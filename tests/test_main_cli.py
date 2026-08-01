from pathlib import Path

import main


def test_parse_cli_defaults_to_generic_metadata_file() -> None:
    args = main._parse_cli([])

    assert args.config == Path("main_metadata.yaml")


def test_parse_cli_accepts_metadata_path() -> None:
    args = main._parse_cli(["--config", "configs/experiment.yaml"])

    assert args.config == Path("configs/experiment.yaml")
