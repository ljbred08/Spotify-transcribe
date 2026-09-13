"""Sampling-parameter settings: defaults, mutual-exclusion lock, validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.config as cfg
from app.settings import Settings


def fresh_settings(monkeypatch=None):
    # isolate from any real settings.json on disk
    d = Path(".test-settings-data")
    d.mkdir(parents=True, exist_ok=True)
    old = cfg.DATA_DIR
    cfg.DATA_DIR = d
    try:
        return Settings()
    finally:
        cfg.DATA_DIR = old


def test_defaults_are_greedy_pair():
    s = fresh_settings().get()
    assert s["temperature"] == 0.0
    assert s["top_p"] == 1.0


def test_temperature_change_snaps_top_p_to_one():
    st = fresh_settings()
    st.update(temperature=0.7)
    assert st.get()["temperature"] == 0.7
    assert st.get()["top_p"] == 1.0


def test_top_p_change_snaps_temperature_to_one():
    st = fresh_settings()
    st.update(top_p=0.9)
    assert st.get()["top_p"] == 0.9
    assert st.get()["temperature"] == 1.0


def test_pair_can_never_both_be_active():
    st = fresh_settings()
    st.update(temperature=1.2, top_p=0.8)   # single payload, both set
    got = st.get()
    # last writer (top_p) wins the active slot; temperature snaps to neutral
    assert got["top_p"] == 0.8 and got["temperature"] == 1.0


def test_neutral_writes_dont_disturb_the_other():
    st = fresh_settings()
    st.update(temperature=0.5)     # active temp, top_p 1
    st.update(top_p=1.0)           # putting top_p back to neutral: temp stays
    assert st.get()["temperature"] == 0.5


def test_range_validation():
    st = fresh_settings()
    for bad in ({"temperature": 3.0}, {"temperature": -0.1},
                {"top_p": 1.5}, {"top_p": 0.01}):
        try:
            st.update(**bad)
            raise AssertionError(f"{bad} should have raised")
        except ValueError:
            pass
