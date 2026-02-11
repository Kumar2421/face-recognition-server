import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _Job:
    fn: Callable[[], Any]
    done: threading.Event
    enqueued_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: BaseException | None = None


class GPUInferenceManager:
    def __init__(
        self,
        *,
        embedder: Any,
        max_queue: int = 256,
        batch_window_ms: int = 0,
    ) -> None:
        self._embedder = embedder
        self._queue: queue.Queue[_Job] = queue.Queue(maxsize=max(1, int(max_queue)))
        self._batch_window_s = max(0.0, float(batch_window_ms) / 1000.0)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-infer", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(
                _Job(fn=lambda: None, done=threading.Event(), enqueued_at=time.time())
            )
        except Exception:
            pass
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _job_timeout_s(self) -> float:
        try:
            return float(os.environ.get("GPU_JOB_TIMEOUT_SEC", "180") or "180")
        except Exception:
            return 180.0

    def _submit(self, fn: Callable[[], Any]) -> Any:
        done = threading.Event()
        j = _Job(fn=fn, done=done, enqueued_at=time.perf_counter())
        try:
            self._queue.put(j, timeout=float(os.environ.get("GPU_QUEUE_PUT_TIMEOUT_SEC", "1") or "1"))
        except queue.Full as e:
            raise RuntimeError("gpu queue full") from e

        timeout_s = self._job_timeout_s()
        done.wait(timeout=timeout_s)
        if not done.is_set():
            raise RuntimeError(
                f"gpu job timed out after {timeout_s:.0f}s (first-run TensorRT engine build may take longer; "
                "increase GPU_JOB_TIMEOUT_SEC)"
            )
        if j.error is not None:
            raise j.error
        return j.result

    def _submit_timed(self, fn: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
        done = threading.Event()
        j = _Job(fn=fn, done=done, enqueued_at=time.perf_counter())
        try:
            self._queue.put(j, timeout=float(os.environ.get("GPU_QUEUE_PUT_TIMEOUT_SEC", "1") or "1"))
        except queue.Full as e:
            raise RuntimeError("gpu queue full") from e

        timeout_s = self._job_timeout_s()
        done.wait(timeout=timeout_s)
        if not done.is_set():
            raise RuntimeError(
                f"gpu job timed out after {timeout_s:.0f}s (first-run TensorRT engine build may take longer; "
                "increase GPU_JOB_TIMEOUT_SEC)"
            )
        if j.error is not None:
            raise j.error

        queue_wait_s = 0.0
        exec_s = 0.0
        try:
            if j.started_at is not None:
                queue_wait_s = max(0.0, float(j.started_at) - float(j.enqueued_at))
            if (j.started_at is not None) and (j.finished_at is not None):
                exec_s = max(0.0, float(j.finished_at) - float(j.started_at))
        except Exception:
            queue_wait_s = 0.0
            exec_s = 0.0

        return j.result, {
            "queue_wait_ms": queue_wait_s * 1000.0,
            "exec_ms": exec_s * 1000.0,
        }

    def detect_best(self, bgr: Any) -> Any:
        return self._submit(lambda: self._embedder.detect_best(bgr))

    def detect_best_timed(self, bgr: Any) -> tuple[Any, dict[str, float]]:
        return self._submit_timed(lambda: self._embedder.detect_best(bgr))

    def detect_all(self, bgr: Any) -> list[Any]:
        return list(self._submit(lambda: self._embedder.detect_all(bgr)))

    def detect_all_timed(self, bgr: Any) -> tuple[list[Any], dict[str, float]]:
        out, t = self._submit_timed(lambda: self._embedder.detect_all(bgr))
        return list(out), t

    def embed_bgr(self, bgr: Any) -> Any:
        return self._submit(lambda: self._embedder.embed_bgr(bgr))

    def embed_bgr_timed(self, bgr: Any) -> tuple[Any, dict[str, float]]:
        return self._submit_timed(lambda: self._embedder.embed_bgr(bgr))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                j = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Optional micro-batch window: wait a bit to allow other jobs to queue.
            # Note: With InsightFace FaceAnalysis.get() we still execute jobs sequentially;
            # this only smooths bursts and avoids request-side contention.
            if self._batch_window_s > 0:
                try:
                    time.sleep(self._batch_window_s)
                except Exception:
                    pass

            try:
                try:
                    j.started_at = time.perf_counter()
                except Exception:
                    j.started_at = None
                j.result = j.fn()
            except BaseException as e:
                j.error = e
            finally:
                try:
                    j.finished_at = time.perf_counter()
                except Exception:
                    j.finished_at = None
                try:
                    j.done.set()
                except Exception:
                    pass
                try:
                    self._queue.task_done()
                except Exception:
                    pass
