"""In-process job runner.

A thread pool rather than Celery/Redis: this is a local desktop-style tool, and
the work is dominated by subprocesses (ffmpeg) and HTTP calls (Ollama) that
release the GIL. Adding a broker would be infrastructure without benefit.

The queue owns project *runs*. Progress is written to the database so the API can
report real state, and cancellation is cooperative via a per-project flag.
"""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

from ..config import get_settings
from ..models.db import Job, Project, get_session
from .states import JobStatus, ProjectStatus

log = logging.getLogger(__name__)


class CancelledError(Exception):
    """Raised inside a run when the user cancels the project."""


class JobQueue:
    """Runs one pipeline per project, with cooperative cancellation."""

    def __init__(self, workers: Optional[int] = None) -> None:
        settings = get_settings()
        self._executor = ThreadPoolExecutor(
            max_workers=workers or settings.job_workers,
            thread_name_prefix="pipeline",
        )
        self._futures: dict[str, Future] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    # --- submission ---------------------------------------------------------
    def submit(self, project_id: str, fn: Callable[[str], None]) -> bool:
        """Queue a pipeline run. Returns False if one is already in flight."""
        with self._lock:
            existing = self._futures.get(project_id)
            if existing is not None and not existing.done():
                return False
            self._cancelled.discard(project_id)
            future = self._executor.submit(self._run, project_id, fn)
            self._futures[project_id] = future
            return True

    def _run(self, project_id: str, fn: Callable[[str], None]) -> None:
        try:
            fn(project_id)
        except CancelledError:
            log.info("Project %s cancelled", project_id)
            self._mark_cancelled(project_id)
        except Exception as exc:  # noqa: BLE001 - the worker must never die silently
            log.exception("Pipeline crashed for project %s", project_id)
            self._mark_failed(project_id, exc)

    # --- cancellation -------------------------------------------------------
    def cancel(self, project_id: str) -> bool:
        with self._lock:
            future = self._futures.get(project_id)
            if future is None or future.done():
                return False
            self._cancelled.add(project_id)
            # Cancel only helps if it has not started; running work exits at the
            # next checkpoint.
            future.cancel()
            return True

    def is_cancelled(self, project_id: str) -> bool:
        with self._lock:
            return project_id in self._cancelled

    def raise_if_cancelled(self, project_id: str) -> None:
        if self.is_cancelled(project_id):
            raise CancelledError(project_id)

    def is_running(self, project_id: str) -> bool:
        with self._lock:
            future = self._futures.get(project_id)
            return future is not None and not future.done()

    # --- failure bookkeeping ------------------------------------------------
    def _mark_failed(self, project_id: str, exc: Exception) -> None:
        message = str(exc) or exc.__class__.__name__
        stage = getattr(exc, "stage", "") or ""
        remedy = getattr(exc, "remedy", "") or ""

        with get_session() as session:
            project = session.get(Project, project_id)
            if project is None:
                return
            project.status = ProjectStatus.FAILED.value
            project.error_message = message[:1000]
            project.error_stage = stage[:64]
            project.error_remedy = remedy[:500]

            for job in project.jobs:
                if job.status == JobStatus.RUNNING.value:
                    job.status = JobStatus.FAILED.value
                    job.error = message[:1000]
            session.commit()

        log.error("Project %s failed: %s\n%s", project_id, message, traceback.format_exc())

    def _mark_cancelled(self, project_id: str) -> None:
        with get_session() as session:
            project = session.get(Project, project_id)
            if project is None:
                return
            project.status = ProjectStatus.CANCELLED.value
            for job in project.jobs:
                if job.status in (JobStatus.RUNNING.value, JobStatus.PENDING.value):
                    job.status = JobStatus.CANCELLED.value
            session.commit()

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


# Module-level singleton used by the API layer.
queue = JobQueue()


class JobReporter:
    """Writes job state to the database so the UI reports real progress.

    Deliberately chatty: the processing screen is only honest if these values
    come from the work actually happening.
    """

    def __init__(self, project_id: str, job_id: str) -> None:
        self.project_id = project_id
        self.job_id = job_id

    def start(self, message: str = "") -> None:
        self._update(status=JobStatus.RUNNING, progress=0.0, message=message, started=True)

    def progress(self, fraction: float, message: str = "") -> None:
        self._update(progress=max(0.0, min(1.0, fraction)), message=message)

    def complete(self, message: str = "") -> None:
        self._update(status=JobStatus.COMPLETED, progress=1.0, message=message, finished=True)

    def skip(self, message: str) -> None:
        self._update(status=JobStatus.SKIPPED, progress=1.0, message=message, finished=True)

    def fail(self, error: str) -> None:
        self._update(status=JobStatus.FAILED, message="", error=error, finished=True)

    def _update(
        self,
        *,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        import datetime as dt

        with get_session() as session:
            job = session.get(Job, self.job_id)
            if job is None:
                return
            if status is not None:
                job.status = status.value
            if progress is not None:
                job.progress = progress
            if message is not None:
                job.message = message[:300]
            if error is not None:
                job.error = error[:2000]
            if started:
                job.started_at = dt.datetime.now(dt.timezone.utc)
                job.attempts += 1
            if finished:
                job.finished_at = dt.datetime.now(dt.timezone.utc)
            session.commit()
