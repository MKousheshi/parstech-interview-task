from pathlib import Path

import pytest
from pydantic import ValidationError

from consultant_bot.common.config import PROJECT_ROOT, Settings


def test_env_file_is_anchored_to_the_project_root_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"
    assert Settings().products_path == PROJECT_ROOT / "products.json"


def test_unknown_search_strategy_is_rejected_up_front(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONSULTANT_BOT_SEARCH_STRATEGY", "bm25")

    with pytest.raises(ValidationError):
        Settings()


def test_prefixed_env_vars_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONSULTANT_BOT_SEARCH_STRATEGY", "tfidf")
    monkeypatch.setenv("CONSULTANT_BOT_SEARCH_TOP_K", "3")

    settings = Settings()

    assert settings.search_strategy == "tfidf"
    assert settings.search_top_k == 3
