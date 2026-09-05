"""Stop AI Clipping Studio.

Normally you just press Ctrl+C in the window `run.bat` opened. This is for the
case where that window is gone, or was never in front of you -- it finds
whatever is listening on the app's port and shuts it down.

  python scripts/stop.py            stop the app
  python scripts/stop.py --ollama   stop Ollama as well

Standard library only, so it works even from a bare Python.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = os.name == "nt"

# Written by launch.py while the app is up.
PID_FILE = ROOT / "data" / ".launcher.pid"


def ok(message: str) -> None:
    print(f"  OK  {message}", flush=True)


def warn(message: str) -> None:
    print(f"  !   {message}", flush=True)


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        path = ROOT / ".env.example"
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k in values})
    return values


def listening_pids(port: int) -> list[int]:
    """PIDs with a LISTENING socket on `port`."""
    pids: list[int] = []
    if WINDOWS:
        output = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True
        ).stdout
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3] == "LISTENING":
                if parts[1].endswith(f":{port}"):
                    pids.append(int(parts[4]))
    else:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        )
        pids = [int(p) for p in result.stdout.split()]
    return sorted(set(pids))


def process_name(pid: int) -> str:
    if not WINDOWS:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True
        )
        return result.stdout.strip()
    output = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not output or output.startswith("INFO:"):
        return ""
    return output.split(",")[0].strip('"')


def alive(pid: int) -> bool:
    return bool(process_name(pid))


def terminate(pid: int) -> bool:
    """Ask nicely, then insist. Returns True once the process is gone."""
    if WINDOWS:
        # /T takes the children too: uvicorn's reloader, and any ffmpeg still
        # rendering underneath it.
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T"],
            capture_output=True,
        )
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True

    for _ in range(10):
        if not alive(pid):
            return True
        time.sleep(0.5)

    if WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True
        )
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True

    time.sleep(1.0)
    return not alive(pid)


def stop_port(port: int, label: str, expect: tuple[str, ...]) -> bool:
    sentence = label[0].upper() + label[1:]
    pids = listening_pids(port)
    if not pids:
        ok(f"{sentence} is not running")
        return True

    stopped = False
    for pid in pids:
        name = process_name(pid)
        # Never kill an unrelated program that happens to hold the port.
        if expect and not any(e in name.lower() for e in expect):
            warn(f"Port {port} is held by {name} (PID {pid}), not {label}. Left alone.")
            continue
        if terminate(pid):
            ok(f"{sentence} stopped ({name}, PID {pid})")
            stopped = True
        else:
            warn(f"Could not stop {name} (PID {pid}).")
    return stopped


def stop_launcher() -> bool:
    """Stop run.bat's launcher, which takes the server down with it.

    Killing only the server would leave the launcher reporting a crashed
    server, and the window run.bat opened waiting for a keypress.
    """
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return False

    name = process_name(pid)
    if "python" not in name.lower():
        PID_FILE.unlink(missing_ok=True)  # stale file from a previous run
        return False

    stopped = terminate(pid)
    if stopped:
        PID_FILE.unlink(missing_ok=True)
    return stopped


def main() -> int:
    env = read_env()
    port = int(env.get("PORT", "8000"))

    print("\n  Stopping AI Clipping Studio\n", flush=True)

    stopped = stop_launcher()
    if stopped:
        ok("The app stopped")
        time.sleep(1.0)

    # Either a leftover server with no launcher, or nothing at all.
    if listening_pids(port) or not stopped:
        stop_port(port, "the app", ("python", "uvicorn"))

    if "--ollama" in sys.argv:
        ollama_url = env.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_port = urlparse(ollama_url).port or 11434
        if stop_port(ollama_port, "Ollama", ("ollama",)):
            time.sleep(1.5)
            if listening_pids(ollama_port):
                warn(
                    "Ollama came back: its tray app restarts it. "
                    "Quit Ollama from the system tray to keep it down."
                )

    print(flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
