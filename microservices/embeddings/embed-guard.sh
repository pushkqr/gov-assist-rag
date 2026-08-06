#!/bin/bash
# Restart the Infinity embedding container when its memory crosses a threshold.
#
# Infinity leaks under sustained ingestion. One observed run climbed from roughly 3.1GB
# to 5.8GB over seven hours while throughput collapsed alongside it: first the batched
# embedding calls timed out, then even single-item fallbacks did. Nothing logged an
# error, so the only symptom was work getting slower. A restart clears it immediately.
#
# This triggers on memory rather than on a timer. A restart costs several minutes of
# model warmup, during which embedding requests fail, so it should happen when the
# service actually needs it and not on a fixed schedule.
#
# Restarting under an active ingestion is safe: ingestion fails a whole document rather
# than recording one with missing passages, so anything attempted during warmup is left
# unrecorded and picked up by a later run.
#
# Install (done automatically by the CloudFormation template):
#   install -m 0755 embed-guard.sh /usr/local/bin/mimir-embed-guard.sh
#   printf '*/10 * * * * root /usr/local/bin/mimir-embed-guard.sh\n' > /etc/cron.d/mimir-embed-guard
#
# Override the defaults with MIMIR_EMBED_DIR and MIMIR_EMBED_MEM_MB.

set -u

THRESHOLD_MB="${MIMIR_EMBED_MEM_MB:-4500}"
COMPOSE_DIR="${MIMIR_EMBED_DIR:-/home/ubuntu/mimir/microservices/embeddings}"
SERVICE=infinity
LOG=/var/log/mimir-embed-guard.log

log() { echo "$(date -Is) $*" >> "$LOG"; }

cd "$COMPOSE_DIR" 2>/dev/null || { log "compose dir missing: $COMPOSE_DIR"; exit 0; }

CID=$(docker compose ps -q "$SERVICE" 2>/dev/null | head -1)
if [ -z "$CID" ]; then
  log "container not running, nothing to do"
  exit 0
fi

RAW=$(docker stats --no-stream --format '{{.MemUsage}}' "$CID" 2>/dev/null)
if [ -z "$RAW" ]; then
  log "could not read memory usage"
  exit 0
fi

# docker reports "1.935GiB / 7.664GiB"; take the used side and normalise to MB.
MB=$(printf '%s' "$RAW" | python3 -c "
import re, sys
s = sys.stdin.read().split('/')[0].strip()
m = re.match(r'([0-9.]+)\s*([KMGT]i?B)', s)
if not m:
    sys.exit(1)
value, unit = float(m.group(1)), m.group(2)
factor = {'KiB': 1/1024, 'MiB': 1, 'GiB': 1024, 'TiB': 1024*1024,
          'KB': 0.000954, 'MB': 0.954, 'GB': 953.7, 'TB': 953674}
print(int(value * factor.get(unit, 1)))
" 2>/dev/null)

if [ -z "$MB" ]; then
  log "could not parse memory usage: $RAW"
  exit 0
fi

if [ "$MB" -ge "$THRESHOLD_MB" ]; then
  log "memory $MB MB at or above $THRESHOLD_MB MB, restarting $SERVICE"
  if docker compose restart "$SERVICE" >> "$LOG" 2>&1; then
    log "restart issued; model warmup takes several minutes"
  else
    log "restart FAILED"
  fi
else
  log "memory $MB MB below $THRESHOLD_MB MB, no action"
fi
