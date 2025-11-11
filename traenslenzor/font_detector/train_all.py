#!/usr/bin/env python3
"""Script to generate datasets and train models for all fonts."""

import sys
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from traenslenzor.font_detector.font_size_model.data_gen import FONT_CONFIGS, generate_dataset
from traenslenzor.font_detector.font_size_model.train import train_model


def main():
    """Generate data and train models for all available fonts."""
    # Select 5 fonts to train - these are in the HuggingFace model's 49 trained classes
    fonts_to_train = [
        "Roboto-Regular",
        "RobotoMono-Regular",
        "Inter-Regular",
        "Lato-Regular",
        "IBMPlexSans-Regular",
    ]

    print("\nTraining font size models...")

    # Configuration
    n_train = 10000
    n_val = 1000
    n_test = 1000

    # Use paths relative to this script
    script_dir = Path(__file__).parent
    data_dir = str(script_dir / "data")
    checkpoints_dir = str(script_dir / "checkpoints")

    results = []

    for font_name in fonts_to_train:
        print(f"\n[{font_name}]")

        # Check if font is available
        if font_name not in FONT_CONFIGS:
            print("  Skipping - not in font config")
            continue

        # Check if font file exists (try all paths)
        font_path = None
        for path in FONT_CONFIGS[font_name]:
            if Path(path).exists():
                font_path = path
                break

        if font_path is None:
            print("  Skipping - font file not found")
            continue

        print(f"  Font: {font_path}")

        try:
            # Generate dataset
            print("  Generating dataset...")
            dataset_stats = generate_dataset(
                font_name=font_name,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                seed=42,
                output_dir=data_dir,
            )

            # Train model
            print("  Training model...")
            training_results = train_model(
                font_name=font_name,
                data_dir=data_dir,
                output_dir=checkpoints_dir,
                epochs=100,
                batch_size=32,
                lr=0.001,
                early_stopping_patience=10,
                verbose=True,
            )

            results.append(
                {
                    "font_name": font_name,
                    "dataset_stats": dataset_stats,
                    "training_results": training_results,
                }
            )

            print(f"  Done (MAE: {training_results['test_mae']:.2f} pt)")

        except Exception as e:
            print(f"  Failed: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n\nTraining Summary:")
    print("-" * 56)

    if not results:
        print("No models were trained successfully.")
        return

    print(f"{'Font':<24} {'MAE':<10} {'RMSE':<10} {'Epoch':<8}")
    print("-" * 56)

    for result in results:
        result_font_name: str = str(result["font_name"])
        train_res: dict[str, Any] = dict(result["training_results"])
        print(
            f"{result_font_name:<24} {train_res['test_mae']:>6.2f} pt  {train_res['test_rmse']:>6.2f} pt  {train_res['best_epoch']:>4}"
        )

    # Overall statistics
    maes = [dict(r["training_results"])["test_mae"] for r in results]
    rmses = [dict(r["training_results"])["test_rmse"] for r in results]

    print("-" * 56)
    print(f"{'Average':<24} {sum(maes) / len(maes):>6.2f} pt  {sum(rmses) / len(rmses):>6.2f} pt")

    print(f"\nModels saved to: {checkpoints_dir}/\n")

    # Save results summary
    import json

    summary_path = Path(checkpoints_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
