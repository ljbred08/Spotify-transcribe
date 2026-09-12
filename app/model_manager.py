"""Catalog of local ONNX ASR models (via onnx-asr) + download management.

Every model lives in its own directory under DATA_DIR/models/<id>/ so that
"delete" is a directory removal and "downloaded" is a directory check.
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import app.config as cfg


@dataclass(frozen=True)
class ModelSpec:
    id: str            # short id used in settings
    repo: str          # huggingface repo id (onnx-asr model name)
    family: str        # parakeet | canary | gigaam | whisper | vosk
    size_mb: int       # approximate download size
    notes: str         # shown in the UI
    languages: str     # short language summary


CATALOG: list[ModelSpec] = [
    ModelSpec("parakeet-v3", "nemo-parakeet-tdt-0.6b-v3", "parakeet",
              2400, "Best quality, fast TDT decoder, word timestamps",
              "25 European languages"),
    ModelSpec("parakeet-v2", "nemo-parakeet-tdt-0.6b-v2", "parakeet",
              2400, "English-only predecessor of v3, word timestamps",
              "English"),
    ModelSpec("canary-1b-flash", "istupakov/canary-1b-flash-onnx", "canary",
              2600, "Spoken word (podcasts) — weak on singing/music",
              "en, de, es, fr"),
    ModelSpec("canary-180m-flash", "istupakov/canary-180m-flash-onnx", "canary",
              780, "Tiny spoken-word model — weak on singing/music",
              "en, de, es, fr"),
    ModelSpec("parakeet-ctc-0.6b", "nemo-parakeet-ctc-0.6b", "parakeet",
              2400, "CTC decoder variant (English), word timestamps",
              "English"),
    ModelSpec("gigaam-v3-rnnt", "gigaam-v3-rnnt", "gigaam",
              1600, "Russian-focused RNNT model",
              "Russian"),
]

_BY_ID = {s.id: s for s in CATALOG}


def get_spec(model_id: str) -> ModelSpec | None:
    return _BY_ID.get(model_id)


def models_dir() -> Path:
    d = cfg.DATA_DIR / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_path(model_id: str) -> Path:
    return models_dir() / model_id


def is_downloaded(model_id: str) -> bool:
    d = local_path(model_id)
    return d.exists() and any(d.iterdir())


def dir_size_mb(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) // (1024 * 1024)


def list_models() -> list[dict]:
    out = []
    for spec in CATALOG:
        downloaded = is_downloaded(spec.id)
        out.append({
            "id": spec.id, "family": spec.family, "notes": spec.notes,
            "languages": spec.languages, "sizeMb": spec.size_mb,
            "downloaded": downloaded,
            "sizeOnDiskMb": dir_size_mb(local_path(spec.id)) if downloaded else None,
        })
    return out


def delete_model(model_id: str) -> bool:
    if not is_downloaded(model_id):
        return False
    shutil.rmtree(local_path(model_id), ignore_errors=True)
    return True


class DownloadManager:
    """Tracks model download jobs and runs them in background threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def start(self, model_id: str) -> dict:
        spec = _BY_ID[model_id]
        with self._lock:
            job = self._jobs.get(model_id)
            if job and job["status"] == "downloading":
                return job
            job = {"modelId": model_id, "status": "downloading",
                   "startedAt": time.time(), "error": None}
            self._jobs[model_id] = job
        t = threading.Thread(target=self._run, args=(spec,), daemon=True)
        t.start()
        return job

    def _run(self, spec: ModelSpec) -> None:
        try:
            import onnx_asr
            # load_model(repo, path) downloads into path (created if missing)
            onnx_asr.load_model(spec.repo, str(local_path(spec.id)))
            with self._lock:
                self._jobs[spec.id] = {
                    "modelId": spec.id, "status": "done",
                    "startedAt": self._jobs[spec.id]["startedAt"],
                    "error": None,
                }
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            with self._lock:
                self._jobs[spec.id] = {
                    "modelId": spec.id, "status": "error",
                    "startedAt": self._jobs[spec.id]["startedAt"],
                    "error": str(exc)[:300],
                }

    def status(self, model_id: str | None = None) -> dict | list:
        with self._lock:
            if model_id:
                return self._jobs.get(model_id, {"modelId": model_id,
                                                 "status": "idle", "error": None})
            return list(self._jobs.values())
