from pathlib import Path

from traenslenzor.doc_classifier.run import load_experiment_config


def test_load_experiment_config_applies_cli_overrides() -> None:
    config = load_experiment_config(
        [
            "--seed",
            "null",
            "--verbose",
            "false",
            "--trainer_config.use_wandb",
            "false",
            "--trainer_config.max_epochs",
            "1",
        ]
    )

    assert config.seed is None
    assert config.verbose is False
    assert config.trainer_config.use_wandb is False
    assert config.trainer_config.max_epochs == 1


def test_load_experiment_config_loads_toml_and_overrides() -> None:
    config = load_experiment_config(
        [
            "--config_path",
            str(Path("_config/cli_test.toml")),
            "--seed",
            "null",
            "--verbose",
            "false",
            "--trainer_config.use_wandb",
            "false",
            "--trainer_config.max_epochs",
            "3",
        ]
    )

    assert config.run_name == "alexnet-train-01"
    assert config.seed is None
    assert config.trainer_config.use_wandb is False
    assert config.trainer_config.max_epochs == 3
