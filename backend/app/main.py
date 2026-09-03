"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import clips, export, projects, system
from .config import get_settings
from .jobs.queue import queue
from .jobs.states import PipelineError
from .models.db import init_db

log = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    # Generated hooks and captions contain emoji; a cp1252 console would raise
    # UnicodeEncodeError from inside the logging call and take down the worker.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty at INFO and drown out pipeline logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    log.info("AI Clipping Studio ready. Data directory: %s", settings.data_dir)
    yield
    queue.shutdown(wait=False)


app = FastAPI(
    title="AI Clipping Studio",
    description="Turns one long video into ready-to-review short-form clips.",
    version="1.0.0",
    lifespan=lifespan,
)

_settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_: Request, exc: PipelineError) -> JSONResponse:
    """Surface actionable errors instead of a generic failure."""
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message, "stage": exc.stage, "remedy": exc.remedy},
    )


api = FastAPI(title="AI Clipping Studio API", version="1.0.0")
api.include_router(system.router)
api.include_router(projects.router)
api.include_router(clips.router)
api.include_router(export.router)
api.add_exception_handler(PipelineError, pipeline_error_handler)

app.mount("/api", api)


# --- static frontend --------------------------------------------------------
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if _frontend_dist.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=_frontend_dist / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        """Serve the built SPA, falling back to index.html for client routes."""
        candidate = _frontend_dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_frontend_dist / "index.html")

else:

    @app.get("/", include_in_schema=False)
    async def dev_notice() -> JSONResponse:
        return JSONResponse(
            {
                "message": "API is running. The frontend has not been built yet.",
                "api_docs": "/api/docs",
                "build_frontend": "cd frontend && npm install && npm run build",
            }
        )
