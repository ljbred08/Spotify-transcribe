"""Cloud catalog integrity + settings migration for renamed model ids."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cloud_models import CATALOG, get_model, _MIGRATE, formatter_catalog


def test_openrouter_ids_are_current():
    ids = {m.id for m in CATALOG}
    assert "openrouter:xiaomi/mimo-v2.5" in ids
    assert "openrouter:thinkingmachines/inkling" in ids
    assert "openrouter:thinkingmachines/inkling-small" in ids
    assert "openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free" in ids
    assert "openrouter:mistralai/voxtral-small-24b-2507" in ids
    # retired slugs are gone
    assert not any("mimo-v2-omni" in i for i in ids)
    assert not any("thinking-machines/" in i for i in ids)


def test_renamed_ids_migrate():
    assert get_model("openrouter:xiaomi/mimo-v2-omni").api_model == "xiaomi/mimo-v2.5"
    assert get_model("openrouter:thinking-machines/inkling").api_model \
        == "thinkingmachines/inkling"
    assert get_model("openrouter:nvidia/nemotron-3-nano-omni").api_model \
        == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    # every migration target exists in the catalog
    for target in _MIGRATE.values():
        assert get_model(target) is not None, target


def test_formatter_catalog_served_per_provider():
    cat = formatter_catalog()
    assert "ollama" in cat and cat["ollama"] == []
    or_models = [m["api_model"] for m in cat["openrouter"]]
    assert "xiaomi/mimo-v2.5" in or_models
    assert "mistralai/voxtral-small-24b-2507" in or_models
    assert all("name" in m for m in cat["gemini"])
