import pytest


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Keep test results out of the real shared cache."""
    monkeypatch.setenv("CONJUNCTIONS_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"
