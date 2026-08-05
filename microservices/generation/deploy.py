#!/usr/bin/env python3
"""Generation service (Ollama): standalone deploy script.

Standard library only, deliberately. This directory is meant to work when copied alone to a
machine that has never seen the rest of this repository:

    scp -r microservices/generation/ user@node:~/mimir-generation/
    ssh user@node 'cd ~/mimir-generation && cp .env.example .env && python3 deploy.py up'

So it cannot import anything from core/ or depend on a package installed system-wide beyond
Docker itself.

This is the one service where hardware genuinely changes which model runs, per this table
(see scratch/build-plan.md Phase 5 for the reasoning):

    CPU, < 15GB RAM   -> qwen3:1.7b
    CPU, >= 15GB RAM  -> qwen3:4b
    GPU, 8-16GB VRAM  -> qwen3:8b
    GPU, > 16GB VRAM  -> qwen3:30b

Point the application at this service with GEN_PROVIDER=local, LOCAL_GEN_URL and
LOCAL_GEN_MODEL matching whatever `deploy.py tier` reports (core/utils.py:local_generate_stream
speaks the standard /v1/chat/completions dialect, so any OpenAI-compatible server works, not
only Ollama).

    python deploy.py check   # is docker available, is .env present
    python deploy.py tier    # report detected hardware and the model tier it implies
    python deploy.py up      # start the container, then pull the tiered model
    python deploy.py down
    python deploy.py status
"""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE_NAME = "generation"
HEALTH_URL = "http://127.0.0.1:11434/api/tags"

GPU_VRAM_THRESHOLD_LARGE_MB = 16000
GPU_VRAM_THRESHOLD_SMALL_MB = 8000
# Nominally-16GB cloud instances (e.g. AWS t3.xlarge) report less than 16 to sysconf due to
# kernel/hypervisor reservation, so a strict >= 16 check misses them. 15 gives headroom.
RAM_THRESHOLD_GB = 15


def _run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=HERE, **kwargs)


def _docker_available():
    if shutil.which("docker") is None:
        return False, "docker CLI not found on PATH. Install Docker first."
    result = _run(["docker", "info"], capture_output=True, text=True)
    if result.returncode != 0:
        return False, "docker daemon not reachable. Is Docker running?"
    return True, "ok"


def _env_ready():
    env_file = HERE / ".env"
    if not env_file.exists():
        return False, f"{env_file.name} missing. Copy .env.example to .env."
    return True, "ok"


def _http_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def detect_ram_gb():
    """Best-effort, stdlib only. Returns None rather than guessing if detection fails on an
    unfamiliar platform - the caller then falls back to the conservative CPU-small tier."""
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError):
        pass
    try:
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / (1024 ** 3)
    except (AttributeError, OSError):
        pass
    return None


def detect_gpu_vram_mb():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def detect_model_tier():
    vram = detect_gpu_vram_mb()
    ram = detect_ram_gb()

    if vram is not None and vram > GPU_VRAM_THRESHOLD_LARGE_MB:
        return "qwen3:30b", f"GPU with {vram} MB VRAM (> {GPU_VRAM_THRESHOLD_LARGE_MB} MB)"
    if vram is not None and vram >= GPU_VRAM_THRESHOLD_SMALL_MB:
        return "qwen3:8b", f"GPU with {vram} MB VRAM ({GPU_VRAM_THRESHOLD_SMALL_MB}-{GPU_VRAM_THRESHOLD_LARGE_MB} MB)"

    ram_desc = f"{ram:.1f} GB RAM" if ram is not None else "RAM undetermined, assuming the conservative tier"
    if ram is not None and ram >= RAM_THRESHOLD_GB:
        return "qwen3:4b", f"CPU only, {ram_desc} (>= {RAM_THRESHOLD_GB} GB)"
    return "qwen3:1.7b", f"CPU only, {ram_desc}"


def _ensure_model_in_env():
    env_file = HERE / ".env"
    text = env_file.read_text(encoding="utf-8")
    if "GEN_MODEL=" in text and "GEN_MODEL=qwen3:4b" not in text:
        return  # operator already set something else; leave it alone
    tier_model, reason = detect_model_tier()
    print(f"Generation model tier: {tier_model} ({reason})")
    if "GEN_MODEL=" in text:
        lines = [f"GEN_MODEL={tier_model}" if line.startswith("GEN_MODEL=") else line
                 for line in text.splitlines()]
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        with env_file.open("a", encoding="utf-8") as f:
            f.write(f"\nGEN_MODEL={tier_model}\n")
    return tier_model


def _read_env_model():
    env_file = HERE / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEN_MODEL="):
            return line.split("=", 1)[1].strip()
    return None


def cmd_check(_args):
    ok_docker, msg_docker = _docker_available()
    print(f"docker      : {'OK' if ok_docker else 'FAIL'} - {msg_docker}")
    ok_env, msg_env = _env_ready()
    print(f".env        : {'OK' if ok_env else 'FAIL'} - {msg_env}")
    return 0 if (ok_docker and ok_env) else 1


def cmd_tier(_args):
    tier_model, reason = detect_model_tier()
    print(f"model tier : {tier_model}")
    print(f"reason     : {reason}")
    return 0


def cmd_up(_args):
    ok_docker, msg_docker = _docker_available()
    if not ok_docker:
        print(f"Cannot start: {msg_docker}")
        return 1
    ok_env, msg_env = _env_ready()
    if not ok_env:
        print(f"Cannot start: {msg_env}")
        return 1

    _ensure_model_in_env()
    model = _read_env_model() or "qwen3:4b"

    print(f"Starting {SERVICE_NAME}...")
    result = _run(["docker", "compose", "up", "-d"])
    if result.returncode != 0:
        return result.returncode

    print("Waiting for the API to come up...", end="", flush=True)
    for _ in range(60):
        if _http_ok(HEALTH_URL):
            print(" ready.")
            break
        print(".", end="", flush=True)
        time.sleep(2)
    else:
        print()
        print(f"Started, but the API did not come up within the timeout. Check: docker compose logs {SERVICE_NAME}")
        return 1

    print(f"Pulling {model} (this can take a while on first run)...")
    pull = _run(["docker", "compose", "exec", "-T", SERVICE_NAME, "ollama", "pull", model])
    if pull.returncode != 0:
        print(f"Model pull failed. The service is up, but {model} is not available yet; "
              f"retry manually with: docker compose exec {SERVICE_NAME} ollama pull {model}")
        return pull.returncode
    print(f"{model} ready. Point the application at GEN_PROVIDER=local, "
          f"LOCAL_GEN_URL=http://<this-host>:11434/v1, LOCAL_GEN_MODEL={model}")
    return 0


def cmd_down(_args):
    return _run(["docker", "compose", "down"]).returncode


def cmd_status(_args):
    up = _http_ok(HEALTH_URL)
    print(f"{SERVICE_NAME}: {'up' if up else 'down or unreachable'} ({HEALTH_URL})")
    _run(["docker", "compose", "ps"])
    return 0 if up else 1


def main():
    # See the root deploy.py for why this matters: without it, this script's own print()
    # calls can appear out of order relative to subprocess (docker compose) output.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Verify Docker and .env are ready. Changes nothing.")
    sub.add_parser("tier", help="Report detected hardware and the model tier it implies. Changes nothing.")
    sub.add_parser("up", help="Start the service and pull the tiered model.")
    sub.add_parser("down", help="Stop the service.")
    sub.add_parser("status", help="Check whether it is reachable right now.")
    args = parser.parse_args()

    handlers = {"check": cmd_check, "tier": cmd_tier, "up": cmd_up, "down": cmd_down, "status": cmd_status}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
