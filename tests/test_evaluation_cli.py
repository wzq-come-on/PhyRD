from scripts.evaluation.cli import main


def test_unified_evaluation_cli_is_importable() -> None:
    assert callable(main)
