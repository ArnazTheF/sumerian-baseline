from pathlib import Path

from click.testing import CliRunner

from sumbase.__main__ import cli
from sumbase.config import load_config


def test_hierarchical_config_and_override():
    config = load_config("configs/mt_byt5.yaml", ["train.batch_size=2"])
    assert config.data.source == "cdli_github"
    assert config.model.name == "google/byt5-small"
    assert config.train.batch_size == 2


def test_all_shipped_configs_load():
    for path in Path("configs").rglob("*.yaml"):
        if path.parent.name in {"data", "model", "train", "generation"}:
            continue
        assert load_config(path).task in {"mt", "translit", "ocr"}


def test_cli_prepare_with_override(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "prepare",
            "--config",
            "configs/smoke/mt.yaml",
            "-o",
            f"paths.data_dir={tmp_path.as_posix()}",
            "-o",
            "data.max_train=2",
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    assert len((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 2
