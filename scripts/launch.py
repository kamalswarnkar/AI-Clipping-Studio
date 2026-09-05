"""One-command launcher for AI Clipping Studio.

Brings up everything the app needs, in order, and stops with a useful message
the moment something is actually missing:

  1. Ollama          - started if it is not already listening
  2. Local models    - checked, and pulled after a prompt if absent
  3. The interface   - built on first run
  4. The server      - started in the foreground so Ctrl+C stops it
  5. The browser      - opened once /api/health actually answers

Standard library only, so it runs before anything is installed.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Detach a background child from this console instead of letting it print over
# the server log.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# Where this process records itself so stop.py can find it.
PID_FILE = ROOT / "data" / ".launcher.pid"

# Exit code meaning "something is wrong and the message on screen explains it".
# run.bat holds the window open for exactly this code, so a double-clicked
# launch that fails stays readable, while one that is simply stopped -- by
# Ctrl+C, or by stop.bat killing this process -- closes without a keypress.
SETUP_FAILURE = 3


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
# Plain ASCII: cmd.exe still defaults to a codepage that mangles box drawing
# and arrows, and a launcher that prints mojibake reads as a broken launcher.


def step(message: str) -> None:
    print(f"  ..  {message}", flush=True)


def ok(message: str) -> None:
    print(f"  OK  {message}", flush=True)


def warn(message: str) -> None:
    print(f"  !   {message}", flush=True)


def fail(message: str, *remedy: str):
    print(f"\n  FAILED: {message}\n", flush=True)
    for line in remedy:
        print(f"      {line}", flush=True)
    if remedy:
        print(flush=True)
    sys.exit(SETUP_FAILURE)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def read_env() -> dict[str, str]:
    """Read .env without a dependency on python-dotenv.

    Falls back to .env.example so a fresh clone still launches; the real
    process environment wins over both.
    """
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


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def responds(url: str, timeout: float = 3.0) -> bool:
    """True when the URL answers with a 2xx. Readiness only, not identity."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def wait_until(check, timeout: float, label: str) -> bool:
    """Poll `check` until it returns truthy, printing a dot per second."""
    deadline = time.monotonic() + timeout
    printed = False
    while time.monotonic() < deadline:
        if check():
            if printed:
                print(flush=True)
            return True
        if not printed:
            print(f"  ..  {label}", end="", flush=True)
            printed = True
        print(".", end="", flush=True)
        time.sleep(1.0)
    if printed:
        print(flush=True)
    return False


# ---------------------------------------------------------------------------
# 1. Ollama
# ---------------------------------------------------------------------------


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    # Installed for the current user, which is the Windows default and is not
    # always on PATH inside a fresh shell.
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Ollama/ollama.exe",
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def ensure_ollama(base_url: str) -> str | None:
    """Return the ollama executable, starting the server if needed."""
    exe = find_ollama()

    if get_json(f"{base_url}/api/tags") is not None:
        ok("Ollama is already running")
        return exe

    if exe is None:
        fail(
            "Ollama is not installed, or not where this script can find it.",
            "Install it from https://ollama.com/download, then run this again.",
        )

    step("Starting Ollama...")
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ollama.log"

    with open(log_path, "ab") as log:
        subprocess.Popen(
            [exe, "serve"],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
        )

    if not wait_until(
        lambda: get_json(f"{base_url}/api/tags") is not None, 60, "Waiting for Ollama"
    ):
        fail(
            f"Ollama did not come up on {base_url} within 60 seconds.",
            f"Its output is in {log_path}",
        )

    ok(f"Ollama started ({base_url})")
    return exe


def installed_models(base_url: str) -> set[str]:
    tags = get_json(f"{base_url}/api/tags", timeout=10) or {}
    names = set()
    for model in tags.get("models", []):
        name = model.get("name", "")
        if name:
            names.add(name)
            # "qwen2.5:7b-instruct" and a bare "qwen2.5" both need to match.
            names.add(name.split(":")[0])
    return names


def ensure_models(exe: str | None, base_url: str, wanted: list[str]) -> None:
    have = installed_models(base_url)
    missing = [m for m in wanted if m not in have and m.split(":")[0] not in have]

    if not missing:
        ok(f"Models ready ({', '.join(wanted)})")
        return

    warn(f"Missing model(s): {', '.join(missing)}")
    print("      These are several GB each and only download once.", flush=True)
    try:
        answer = input("      Download them now? [Y/n] ").strip().lower()
    except EOFError:
        answer = "y"
    if answer not in ("", "y", "yes"):
        fail(
            "Cannot run without the models.",
            *[f"ollama pull {name}" for name in missing],
        )

    for name in missing:
        step(f"Pulling {name} (this will take a while)...")
        result = subprocess.run([exe or "ollama", "pull", name])
        if result.returncode != 0:
            fail(f"'ollama pull {name}' failed.")
    ok("Models downloaded")


# ---------------------------------------------------------------------------
# 2. Python environment and interface
# ---------------------------------------------------------------------------


def find_python() -> Path:
    for relative in ("venv/Scripts/python.exe", "venv/bin/python"):
        candidate = ROOT / relative
        if candidate.is_file():
            return candidate
    fail(
        "The 'venv' virtual environment is missing.",
        "python -m venv venv",
        "venv\\Scripts\\python.exe -m pip install -r backend/requirements.txt",
    )


def ensure_dependencies(python: Path) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import uvicorn, fastapi"],
        capture_output=True,
    )
    if probe.returncode != 0:
        fail(
            "The backend dependencies are not installed in venv.",
            f'"{python}" -m pip install -r backend/requirements.txt',
        )
    ok("Python environment ready")


def ensure_interface() -> None:
    if (ROOT / "frontend/dist/index.html").is_file():
        ok("Interface already built")
        return

    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        fail(
            "The interface is not built and npm was not found.",
            "Install Node 18+ from https://nodejs.org, then run this again.",
        )

    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").is_dir():
        step("Installing interface packages (first run only)...")
        if subprocess.run([npm, "install"], cwd=frontend).returncode != 0:
            fail("'npm install' failed.")

    step("Building the interface (first run only)...")
    if subprocess.run([npm, "run", "build"], cwd=frontend).returncode != 0:
        fail("'npm run build' failed.")
    ok("Interface built")


# ---------------------------------------------------------------------------
# 3. Server
# ---------------------------------------------------------------------------


def port_in_use(host: str, port: int) -> bool:
    target = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((target, port)) == 0


def open_browser_when_ready(url: str) -> None:
    """Open the browser only once the app answers, never on a dead port.

    Deliberately polls the page rather than /api/health: health re-probes
    FFmpeg and Ollama on every call and takes ~4.5s, which is fine for a status
    panel and far too slow to poll.
    """
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if responds(url):
            print(f"\n  Opening {url}\n", flush=True)
            webbrowser.open(url)
            return
        time.sleep(0.5)
    warn(f"Server did not answer in time. Open {url} yourself once it does.")


def main() -> int:
    no_browser = "--no-browser" in sys.argv

    env = read_env()
    host = env.get("HOST", "127.0.0.1")
    port = int(env.get("PORT", "8000"))
    ollama_url = env.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    visible_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    app_url = f"http://{visible_host}:{port}"
    health_url = f"{app_url}/api/health"

    print("\n  AI Clipping Studio\n", flush=True)

    # Already running? Just show it rather than failing on a busy port.
    if port_in_use(host, port):
        # Generous timeout: health shells out to FFmpeg and queries Ollama.
        if get_json(health_url, timeout=30) is not None:
            ok(f"Already running at {app_url}")
            if not no_browser:
                webbrowser.open(app_url)
            return 0
        fail(
            f"Port {port} is taken by something else.",
            "Stop that process, or change PORT in .env.",
        )

    exe = ensure_ollama(ollama_url)

    wanted = [env.get("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct")]
    if env.get("VISION_ENABLED", "false").lower() == "true":
        wanted.append(env.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b"))
    ensure_models(exe, ollama_url, wanted)

    python = find_python()
    ensure_dependencies(python)
    ensure_interface()

    if not no_browser:
        threading.Thread(
            target=open_browser_when_ready,
            args=(app_url,),
            daemon=True,
        ).start()

    print(f"\n  Starting the server at {app_url}", flush=True)
    print("  Press Ctrl+C to stop.\n", flush=True)

    server = subprocess.Popen(
        [
            str(python),
            "-m",
            "uvicorn",
            "app.main:app",
            "--app-dir",
            "backend",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=ROOT,
    )

    # stop.py stops this process rather than the server underneath it, so that
    # the window run.bat opened closes instead of reporting a dead server.
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    try:
        code = server.wait()
        if code != 0:
            # Distinct from a plain non-zero exit so run.bat knows to hold the
            # window open and show what went wrong.
            warn(f"The server stopped unexpectedly (exit {code}).")
            return SETUP_FAILURE
        return 0
    except KeyboardInterrupt:
        # Ollama is deliberately left running: it may have been started by the
        # tray app, and a warm model makes the next launch much faster.
        print("\n  Stopping the server. (Ollama is left running.)\n", flush=True)
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        return 0
    finally:
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
