from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_configs():
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        yield path, value


def test_model_configs_use_the_deterministic_registry_contract() -> None:
    for path, config in load_configs():
        if "model" not in config:
            continue
        deterministic = config["model"]["deterministic"]
        assert set(deterministic) == {"name", "params"}, path
        assert isinstance(deterministic["name"], str) and deterministic["name"], path
        assert isinstance(deterministic["params"], dict), path


def test_checkpoint_policy_has_no_periodic_checkpoint_setting() -> None:
    for path, config in load_configs():
        optimization = config.get("optimization", {})
        assert "checkpoint_every" not in optimization, path
        assert "checkpoint_every_epochs" not in optimization, path


def test_residual_runs_consume_a_supported_deterministic_checkpoint() -> None:
    for path, config in load_configs():
        if config.get("stage") != "residual":
            continue
        checkpoint = config["model"].get("deterministic_checkpoint")
        if checkpoint is not None:
            assert checkpoint.endswith(("checkpoint_best.pt", "checkpoint_final.pt")), path


def test_formal_training_configs_enable_validation_for_best_selection() -> None:
    for path, config in load_configs():
        if not path.name.startswith("train_") or "model" not in config:
            continue
        assert config.get("validation", {}).get("enabled") is True, path
