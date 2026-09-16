"""The lyrics font-size override: unset by default, bounded when set, cleared again."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import app.config as cfg
from app.settings import Settings


def fresh_settings(tmp_path):
    """Settings isolated in a per-test tmp dir (no real settings.json)."""
    old = cfg.DATA_DIR
    cfg.DATA_DIR = Path(tmp_path)
    try:
        return Settings()
    finally:
        cfg.DATA_DIR = old


def test_default_leaves_the_layouts_alone(tmp_path):
    # None, not 25: each layout (phone/tablet/desktop/full-window) keeps its own
    # size until the user actually moves the slider
    assert fresh_settings(tmp_path).get()["lyric_font_size"] is None


def test_setting_persists(tmp_path):
    st = fresh_settings(tmp_path)
    st.update(lyric_font_size=21)
    assert st.get()["lyric_font_size"] == 21.0


@pytest.mark.parametrize("value", [None, "", 0, "0"])
def test_clearing_goes_back_to_the_layout_default(tmp_path, value):
    st = fresh_settings(tmp_path)
    st.update(lyric_font_size=31)
    st.update(lyric_font_size=value)
    assert st.get()["lyric_font_size"] is None


@pytest.mark.parametrize("value", [11, 73, -5])
def test_out_of_range_is_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        fresh_settings(tmp_path).update(lyric_font_size=value)


def test_non_numeric_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        fresh_settings(tmp_path).update(lyric_font_size="huge")


def test_bounds_are_inclusive(tmp_path):
    st = fresh_settings(tmp_path)
    st.update(lyric_font_size=12)
    assert st.get()["lyric_font_size"] == 12.0
    st.update(lyric_font_size=72)
    assert st.get()["lyric_font_size"] == 72.0
