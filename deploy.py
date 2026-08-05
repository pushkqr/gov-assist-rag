#!/usr/bin/env python3
"""Mimir deployment orchestrator.

Brings the whole stack up on one machine, for a fully self-hosted deployment (see
microservices/README.md and scratch/build-plan.md Phase 5/6). Runs from a full checkout of
this repository, unlike the scripts under microservices/*/, which are each standalone and
copyable to a machine that has never seen this repo.

Core principle this script follows: it does not reimplement any service's startup logic. It
shells out to the exact same deploy.py each service directory carries on its own — the one
you would run after `scp -r microservices/embeddings/ node:`. One code path, two entry
points. If this orchestrator had its own copy of that logic, the two would drift, and the
remote-node path is the one that would break, silently, exactly when it's needed on stage or
in front of a department that just handed over a server.

    python deploy.py check              # hardware report + can each service reach the network
    python deploy.py up                 # bring every service up, in order, then the app
    python deploy.py up --only weaviate  # just one service
    python deploy.py down                # tear everything down, reverse order
    python deploy.py status              # live reachability, same probes as the admin panel
    python deploy.py logs weaviate       # passthrough to that service's container logs
"""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MICROSERVICES_DIR = REPO_ROOT / "microservices"

# Order matters for `up`/`down` only in the sense that infrastructure should be reachable
# before the application starts; the services themselves have no dependencies on each other.
SERVICE_ORDER = ["weaviate", "embeddings", "translation", "generation"]
OPTIONAL_SERVICES = ["docling"]  # not brought up by default; pass --only docling to include


def _service_dir(name: str) -> Path:
    return MICROSERVICES_DIR / name


def _run_service_script(name: str, command: str) -> int:
    script = _service_dir(name) / "deploy.py"
    if not script.exists():
        print(f"[{name}] no deploy.py found at {script}")
        return 1
    result = subprocess.run([sys.executable, str(script), command])
    return result.returncode


def _detect_ram_gb():
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


def _detect_gpu_vram_mb():
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


def _disk_free_gb():
    try:
        return shutil.disk_usage(REPO_ROOT).free / (1024 ** 3)
    except OSError:
        return None


def cmd_check(_args):
    print("Hardware")
    print("--------")
    ram = _detect_ram_gb()
    vram = _detect_gpu_vram_mb()
    disk = _disk_free_gb()
    print(f"  CPUs        : {os.cpu_count() or 'unknown'}")
    print(f"  RAM         : {f'{ram:.1f} GB' if ram is not None else 'undetermined'}")
    print(f"  GPU VRAM    : {f'{vram} MB' if vram is not None else 'no GPU detected'}")
    print(f"  Disk free   : {f'{disk:.1f} GB' if disk is not None else 'undetermined'} "
          f"(images ~12GB, models ~5GB, plus your corpus index)")

    print()
    print("Per-service readiness (docker + .env), via each service's own `deploy.py check`")
    print("--------------------------------------------------------------------------------")
    all_ok = True
    for name in SERVICE_ORDER:
        print(f"\n[{name}]")
        rc = _run_service_script(name, "check")
        all_ok = all_ok and (rc == 0)
    return 0 if all_ok else 1


def cmd_up(args):
    targets = [args.only] if args.only else list(SERVICE_ORDER)
    unknown = [t for t in targets if t not in SERVICE_ORDER + OPTIONAL_SERVICES]
    if unknown:
        print(f"Unknown service(s): {', '.join(unknown)}. Known: {', '.join(SERVICE_ORDER + OPTIONAL_SERVICES)}")
        return 1

    for name in targets:
        print(f"\n=== {name} ===")
        rc = _run_service_script(name, "up")
        if rc != 0:
            print(f"\n[{name}] failed to come up (exit {rc}). Stopping here rather than "
                  f"starting the application against a stack that isn't fully green.")
            return rc

    if args.only:
        print(f"\n{args.only} is up. Not starting the application (--only was given).")
        return 0

    print("\n=== application ===")
    result = subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=REPO_ROOT)
    if result.returncode != 0:
        return result.returncode

    print("\nAll services and the application are up. Run `python deploy.py status` to verify "
          "the application can actually reach each one (network policy, firewall, etc. can "
          "still block a service that Docker itself reports as running).")
    return 0


def cmd_down(args):
    if not args.only:
        print("=== application ===")
        subprocess.run(["docker", "compose", "down"], cwd=REPO_ROOT)

    targets = [args.only] if args.only else list(reversed(SERVICE_ORDER))
    all_ok = True
    for name in targets:
        print(f"\n=== {name} ===")
        rc = _run_service_script(name, "down")
        all_ok = all_ok and (rc == 0)
    return 0 if all_ok else 1


def cmd_status(_args):
    """Same probes /api/admin/topology uses, run standalone. Requires this repo's Python
    environment (google-genai, weaviate-client, etc.), unlike the per-service `deploy.py
    status` commands, which only need the standard library."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from core.health import run_all_probes
    except ImportError as exc:
        print(f"Could not import this repo's dependencies ({exc}). "
              f"Run this from an environment with requirements.txt installed, "
              f"or check individual services with e.g. `python microservices/weaviate/deploy.py status`.")
        return 1

    result = run_all_probes()
    print(f"{result['self_hosted']} of {len(result['components'])} components self-hosted "
          f"(generation provider: {result['generation_provider']})\n")
    all_up = True
    for component in result["components"]:
        status = component["status"]
        all_up = all_up and (status == "up")
        marker = "UP  " if status == "up" else "DOWN"
        print(f"  [{marker}] {component['name']:<18} {component['host']:<28} "
              f"{component['latency_ms']:>5} ms   {component['detail']}")
    return 0 if all_up else 1


def cmd_logs(args):
    if args.name not in SERVICE_ORDER + OPTIONAL_SERVICES:
        print(f"Unknown service: {args.name}. Known: {', '.join(SERVICE_ORDER + OPTIONAL_SERVICES)}")
        return 1
    return subprocess.run(["docker", "compose", "logs", "-f"], cwd=_service_dir(args.name)).returncode


def main():
    # Without this, this script's own print() calls can appear after a child subprocess's
    # output even though they were written first: Python buffers stdout when it isn't a
    # terminal (e.g. piped, or under this environment's tool wrapper), while the child writes
    # straight to the shared file descriptor. Line-buffering keeps the interleaving honest.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass  # older Python; harmless to skip

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Hardware report and per-service readiness. Changes nothing.")

    up_parser = sub.add_parser("up", help="Bring every service up, in order, then the application.")
    up_parser.add_argument("--only", choices=SERVICE_ORDER + OPTIONAL_SERVICES, help="Bring up just one service.")

    down_parser = sub.add_parser("down", help="Tear everything down, reverse order.")
    down_parser.add_argument("--only", choices=SERVICE_ORDER + OPTIONAL_SERVICES, help="Tear down just one service.")

    sub.add_parser("status", help="Live reachability of every component, same probes as the admin panel.")

    logs_parser = sub.add_parser("logs", help="Follow one service's container logs.")
    logs_parser.add_argument("name", choices=SERVICE_ORDER + OPTIONAL_SERVICES)

    args = parser.parse_args()
    handlers = {"check": cmd_check, "up": cmd_up, "down": cmd_down, "status": cmd_status, "logs": cmd_logs}
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
