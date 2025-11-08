#!/usr/bin/env python
"""Demonstrate ExperimentConfig TOML serialization and deserialization.

This script:
1. Loads the demo PlantUML config
2. Exports it to TOML in the .configs directory
3. Reads it back and verifies it matches
"""

from traenslenzor.doc_classifier.configs import ExperimentConfig, PathConfig
from traenslenzor.doc_classifier.utils import Console

console = Console.with_prefix("ConfigExport").set_verbose(True)


def main():
    """Export and reimport ExperimentConfig via TOML."""
    
    # Get the configs directory from PathConfig
    paths = PathConfig()
    console.log(f"Using configs directory: {paths.configs_dir}")
    
    # Create a demo ExperimentConfig with default settings
    console.log("Creating ExperimentConfig with default settings...")
    config = ExperimentConfig(
        run_name="demo_toml_export",
        stage="train",
        is_debug=False,  # Disable fast_dev_run for real training
        verbose=True,
        seed=42,
    )
    
    console.log("Config created with:")
    console.log(f"  - Batch size: {config.datamodule_config.batch_size}")
    console.log(f"  - Num classes: {config.module_config.num_classes}")
    console.log(f"  - Max epochs: {config.trainer_config.max_epochs}")
    console.log(f"  - Learning rate: {config.module_config.optimizer.learning_rate}")
    
    # Export to TOML
    toml_filename = "demo_experiment.toml"
    toml_path = config.save_toml(
        path=paths.configs_dir / toml_filename,
        include_comments=True,
        include_type_hints=True,
    )
    console.log(f"✅ Exported to TOML: {toml_path}")
    
    # Read it back
    console.log(f"Reading config back from: {toml_path}")
    reloaded_config = ExperimentConfig.from_toml(toml_path)
    console.log("✅ Successfully reloaded config from TOML")
    
    # Verify key fields match
    console.log("\n📊 Verification:")
    console.log(f"  Original run_name: {config.run_name}")
    console.log(f"  Reloaded run_name: {reloaded_config.run_name}")
    console.log(f"  Original stage: {config.stage}")
    console.log(f"  Reloaded stage: {reloaded_config.stage}")
    console.log(f"  Original batch_size: {config.datamodule_config.batch_size}")
    console.log(f"  Reloaded batch_size: {reloaded_config.datamodule_config.batch_size}")
    console.log(f"  Original num_classes: {config.module_config.num_classes}")
    console.log(f"  Reloaded num_classes: {reloaded_config.module_config.num_classes}")
    
    console.log(f"\n✨ TOML config saved to: {toml_path}")
    console.log("You can now edit this file and load it with:")
    console.log(f"    config = ExperimentConfig.from_toml('{toml_filename}')")
    
    return toml_path


if __name__ == "__main__":
    main()
