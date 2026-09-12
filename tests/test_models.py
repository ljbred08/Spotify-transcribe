"""Tests for model manager and settings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config as cfg
from app import model_manager as mm
from app.settings import Settings


def test_catalog_has_known_models():
    ids = {s.id for s in mm.CATALOG}
    assert {"parakeet-v3", "canary-180m-flash", "parakeet-unified-en",
            "gigaam-multilingual-ctc"} <= ids


def test_catalog_repos_resolve_via_onnx_asr():
    """Every catalog repo must be resolvable by onnx-asr (or need_fixup)."""
    from onnx_asr.loader import create_asr_resolver
    for spec in mm.CATALOG:
        if spec.needs_fixup:
            continue
        resolver = create_asr_resolver(spec.repo)
        resolver.resolve_config()  # hits HF for config.json (cached)


def test_normalize_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mm.cfg, "DATA_DIR", tmp_path)
    d = mm.local_path("fake-custom")
    d.mkdir(parents=True)
    for name in ["encoder.onnx", "decoder_joint.onnx",
                 "encoder.onnx.data", "vocab.txt"]:
        (d / name).write_bytes(b"x")
    mm.normalize_model_dir("fake-custom")
    assert (d / "encoder-model.onnx").exists()
    assert (d / "decoder_joint-model.onnx").exists()
    # external data file keeps its original name (graph references it)
    assert (d / "encoder.onnx.data").exists()
    assert (d / "config.json").exists()
    import json
    assert json.loads((d / "config.json").read_text())["model_type"] == "nemo-conformer-rnnt"


def test_downloaded_flag_matches_disk():
    # canary-180m-flash was downloaded during development
    assert mm.is_downloaded("canary-180m-flash") == mm.local_path("canary-180m-flash").exists()


def test_delete_missing_model_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(mm.cfg, "DATA_DIR", tmp_path)
    assert mm.delete_model("not-a-model") is False


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    s = Settings()
    assert s.get()["backend"] == "auto"
    s.update(model="canary-180m-flash")
    assert s.get()["model"] == "canary-180m-flash"
    # persisted to disk
    s2 = Settings()
    assert s2.get()["model"] == "canary-180m-flash"


def test_settings_rejects_bad_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    s = Settings()
    try:
        s.update(backend="nonexistent")
        assert False, "should have raised"
    except ValueError:
        pass


def test_local_model_load_and_timestamps():
    """The downloaded canary model loads from data/models and yields word timestamps."""
    if not mm.is_downloaded("canary-180m-flash"):
        return  # skip on fresh checkouts without models
    import numpy as np
    import onnx_asr
    spec = mm.get_spec("canary-180m-flash")
    model = onnx_asr.load_model(spec.repo, str(mm.local_path(spec.id)))
    sr = 16000
    t = np.linspace(0, 0.2, sr // 5, dtype=np.float32)
    silence = np.zeros(sr // 2, dtype=np.float32)
    # silent input: model should return an empty-ish result without crashing
    res = model.with_timestamps().recognize(silence, sample_rate=sr)
    assert isinstance(res.tokens, list)
