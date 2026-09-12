"""Catalog of local ONNX ASR models (via onnx-asr) + download management.

Every model lives in its own directory under DATA_DIR/models/<id>/ so that
"delete" is a directory removal and "downloaded" is a directory check.
"""
from __future__ import annotations

import json
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
    family: str        # parakeet | canary | gigaam
    size_mb: int       # approximate fp32 download size
    notes: str         # shown in the UI
    languages: str     # short language summary
    needs_fixup: bool = False  # repo lacks onnx-asr layout -> normalize after download


# Every English + multilingual model onnx-asr supports that returns word
# timestamps (the whisper family is excluded: onnx-asr's whisper decoding
# discards timestamps entirely, so it cannot drive karaoke timing).
CATALOG: list[ModelSpec] = [
    ModelSpec("parakeet-v3", "nemo-parakeet-tdt-0.6b-v3", "parakeet",
              2500, "Best all-round: fast TDT decoder, handles singing, word timestamps",
              "25 European languages"),
    ModelSpec("parakeet-v2", "nemo-parakeet-tdt-0.6b-v2", "parakeet",
              2500, "English-only predecessor of v3, word timestamps",
              "English"),
    ModelSpec("parakeet-unified-en", "bobNight/parakeet-unified-en-0.6b-onnx", "parakeet",
              2500, "Unified offline/streaming RNNT, built-in punctuation & capitalization (community export)",
              "English", needs_fixup=True),
    ModelSpec("parakeet-ctc-0.6b", "nemo-parakeet-ctc-0.6b", "parakeet",
              2500, "CTC decoder variant (English), word timestamps",
              "English"),
    ModelSpec("parakeet-rnnt-0.6b", "nemo-parakeet-rnnt-0.6b", "parakeet",
              2500, "RNN-T decoder variant (English), word timestamps",
              "English"),
    ModelSpec("canary-1b-v2", "nemo-canary-1b-v2", "canary",
              4000, "Largest Canary: strong on spoken word — weak on singing/music",
              "25 languages (en, de, es, fr, …)"),
    ModelSpec("canary-1b-flash", "istupakov/canary-1b-flash-onnx", "canary",
              3600, "Spoken word (podcasts) — weak on singing/music",
              "en, de, es, fr"),
    ModelSpec("canary-180m-flash", "istupakov/canary-180m-flash-onnx", "canary",
              780, "Tiny spoken-word model — weak on singing/music",
              "en, de, es, fr"),
    ModelSpec("gigaam-multilingual-ctc", "gigaam-multilingual-ctc", "gigaam",
              900, "Compact multilingual CTC, word timestamps",
              "70+ languages (best on CIS)"),
    ModelSpec("gigaam-multilingual-large-ctc", "gigaam-multilingual-large-ctc", "gigaam",
              2350, "Large multilingual CTC, word timestamps",
              "70+ languages (best on CIS)"),
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


# Files renamed when normalizing community exports into the onnx-asr layout.
# NOTE: only the .onnx graphs are renamed; *.onnx.data external-weight files
# keep their original names because the graph embeds the exact data filename.
_FIXUP_RENAMES = {
    "encoder.onnx": "encoder-model.onnx",
    "decoder_joint.onnx": "decoder_joint-model.onnx",
    "encoder.int8.onnx": "encoder-model.int8.onnx",
    "decoder_joint.int8.onnx": "decoder_joint-model.int8.onnx",
}

# NeMo transducer config synthesized for repos that ship none.
_FIXUP_CONFIG = {
    "model_type": "nemo-conformer-rnnt",
    "features_size": 128,
    "subsampling_factor": 8,
    "max_tokens_per_step": 10,
}


def normalize_model_dir(model_id: str) -> None:
    """Make a community-exported repo loadable by onnx-asr.

    Some exports (e.g. bobNight/parakeet-unified-en-0.6b-onnx) ship no
    config.json and use filenames onnx-asr doesn't glob for. Rename the
    graphs to the expected names, synthesize the config, and convert a
    one-token-per-line vocab.txt into onnx-asr's "token id" format.
    """
    d = local_path(model_id)
    for src, dst in _FIXUP_RENAMES.items():
        s, t = d / src, d / dst
        if s.exists() and not t.exists():
            s.rename(t)
    cfg = d / "config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps(_FIXUP_CONFIG, indent=2), encoding="utf-8")
    _fixup_vocab(d)


# Authoritative token map for parakeet-unified-en: the sherpa-onnx export of
# the same NVIDIA model. The community ONNX export strips the SentencePiece
# word-boundary markers (\u2581) and the <unk>/<blk> specials from its
# vocab.txt, which makes ids misaligned and output space-less — so we source
# the true 1025-entry mapping (ids verified against the joint-net output).
_SHERPA_TOKENS_URL = ("https://huggingface.co/csukuangfj2/"
                      "sherpa-onnx-nemo-parakeet-unified-en-0.6b-non-streaming"
                      "/resolve/main/tokens.txt")


def _fixup_vocab(d: Path) -> None:
    """Rebuild vocab.txt with onnx-asr's `<token> <id>` format and <blk>.

    onnx-asr needs a `<blk> <id>` line to locate the transducer's blank class;
    community exports often ship a lossy one-token-per-line vocab instead.
    """
    import urllib.request

    vocab = d / "vocab.txt"
    if vocab.exists():
        lines = vocab.read_text(encoding="utf-8").splitlines()
        if lines and any("<blk>" in ln for ln in lines):
            return  # already usable
    tokens = urllib.request.urlopen(_SHERPA_TOKENS_URL, timeout=60).read().decode("utf-8").strip()
    backup = d / "vocab.tokens.txt"
    if not backup.exists() and vocab.exists():
        backup.write_text(vocab.read_text(encoding="utf-8"), encoding="utf-8")
    vocab.write_text(tokens + "\n", encoding="utf-8")


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
            if spec.needs_fixup:
                # Repo can't go through onnx-asr's loader (no config.json),
                # so pull the fp32 files directly, then normalize the layout.
                from huggingface_hub import snapshot_download

                snapshot_download(
                    spec.repo,
                    local_dir=str(local_path(spec.id)),
                    allow_patterns=["*.onnx", "*.onnx.data", "vocab.txt", "tokenizer.model"],
                    ignore_patterns=["*.int8*"],
                )
                normalize_model_dir(spec.id)
            else:
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
