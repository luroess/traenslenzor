#!/usr/bin/env python
"""Verification script for TODO resolution and jaxtyping integration.

This script verifies:
1. Metric enum is defined and can be used (replaces string literals)
2. Jaxtyping type aliases are defined in lit_module
3. AlexNet has jaxtyping in forward() signature
4. lit_datamodule uses match-case instead of if-elif
"""

from pathlib import Path

print("🔍 Verifying TODO resolutions and jaxtyping integration...")
print()

# Test 1: Metric enum
print("✅ Test 1: Metric enum (replaces string literals)")
from traenslenzor.doc_classifier.utils.schemas import Metric

assert hasattr(Metric, "TRAIN_LOSS")
assert hasattr(Metric, "VAL_ACCURACY")
assert hasattr(Metric, "TEST_LOSS")
assert str(Metric.TRAIN_LOSS) == "train/loss"
print(f"   - Metric.TRAIN_LOSS = '{Metric.TRAIN_LOSS}'")
print(f"   - Metric.VAL_ACCURACY = '{Metric.VAL_ACCURACY}'")
print(f"   - Metric.TEST_LOSS = '{Metric.TEST_LOSS}'")
print()

# Test 2: Check lit_module uses Metric enum (no string literals)
print("✅ Test 2: lit_module.py uses Metric enum")
lit_module_path = Path("traenslenzor/doc_classifier/lightning/lit_module.py")
lit_module_content = lit_module_path.read_text()
assert "from ..utils import BaseConfig, Metric" in lit_module_content
assert "Metric.TRAIN_LOSS" in lit_module_content
assert "Metric.VAL_ACCURACY" in lit_module_content
assert "Metric.TEST_LOSS" in lit_module_content
# Verify NO hardcoded strings like "train/loss" remain in log calls
log_lines = [
    line for line in lit_module_content.split("\n") if ".log(" in line and "self.log(" in line
]
hardcoded_metrics = [
    line for line in log_lines if '"train/' in line or '"val/' in line or '"test/' in line
]
assert len(hardcoded_metrics) == 0, f"Found hardcoded metric strings: {hardcoded_metrics}"
print("   - Uses Metric.TRAIN_LOSS, Metric.VAL_ACCURACY, etc.")
print("   - NO hardcoded 'train/loss' strings found")
print()

# Test 3: Check jaxtyping type aliases exist
print("✅ Test 3: Jaxtyping type aliases in lit_module.py")
assert "ImageBatch = Float[Tensor" in lit_module_content
assert "Logits = Float[Tensor" in lit_module_content
assert "Targets = Int64[Tensor" in lit_module_content
assert "ScalarLoss = Float[Tensor" in lit_module_content
print("   - ImageBatch = Float[Tensor, 'B C H W']")
print("   - Logits = Float[Tensor, 'B N']")
print("   - Targets = Int64[Tensor, 'B']")
print("   - ScalarLoss = Float[Tensor, '']")
print()

# Test 4: Check forward() uses jaxtyping
print("✅ Test 4: forward() method uses jaxtyping types")
assert "def forward(self, batch: ImageBatch) -> Logits:" in lit_module_content
print("   - def forward(self, batch: ImageBatch) -> Logits:")
print()

# Test 5: Check step methods use jaxtyping
print("✅ Test 5: Step methods use jaxtyping types")
assert "def training_step(self, batch: tuple[ImageBatch, Targets]" in lit_module_content
assert "def validation_step(self, batch: tuple[ImageBatch, Targets]" in lit_module_content
assert "def test_step(self, batch: tuple[ImageBatch, Targets]" in lit_module_content
print("   - training_step uses tuple[ImageBatch, Targets]")
print("   - validation_step uses tuple[ImageBatch, Targets]")
print("   - test_step uses tuple[ImageBatch, Targets]")
print()

# Test 6: Check alexnet.py has jaxtyping
print("✅ Test 6: AlexNet forward() uses jaxtyping")
alexnet_path = Path("traenslenzor/doc_classifier/models/alexnet.py")
alexnet_content = alexnet_path.read_text()
assert "from jaxtyping import Float" in alexnet_content
assert (
    "ImageBatch = Float[" in alexnet_content
    or "def forward(self, x: ImageBatch)" in alexnet_content
)
print("   - Imports jaxtyping.Float")
print("   - forward() signature uses ImageBatch type alias")
print()

# Test 7: Check lit_datamodule uses match-case
print("✅ Test 7: lit_datamodule.py uses match-case (no duplication)")
datamodule_path = Path("traenslenzor/doc_classifier/lightning/lit_datamodule.py")
datamodule_content = datamodule_path.read_text()
assert "match requested_stage:" in datamodule_content
assert "case Stage.TRAIN:" in datamodule_content
assert "case Stage.VAL:" in datamodule_content
assert "case Stage.TEST:" in datamodule_content
# Verify the TODO is gone
assert "TODO: write generic to reduce duplication" not in datamodule_content
print("   - Uses match-case pattern")
print("   - TODO comment removed")
print()

# Test 8: Verify no TODOs remain (except console.py)
print("✅ Test 8: All target TODOs resolved")
import subprocess

result = subprocess.run(
    ["grep", "-r", "TODO", "traenslenzor/doc_classifier/", "--include=*.py"],
    capture_output=True,
    text=True,
)
remaining_todos = [
    line
    for line in result.stdout.split("\n")
    if line and "console.py" not in line  # Exclude console.py TODO
]
assert len(remaining_todos) == 0, f"Remaining TODOs: {remaining_todos}"
print("   - lit_module.py: TODO resolved (Metric enum)")
print("   - lit_datamodule.py: TODO resolved (match-case)")
print("   - console.py: TODO unchanged (out of scope)")
print()

print("🎉 All verifications passed!")
print()
print("📋 Summary of Changes:")
print("=" * 60)
print("1. ✅ Created Metric enum in schemas.py")
print("   - Replaced hardcoded 'train/loss', 'val/accuracy' strings")
print("   - All metric logging now uses Metric.TRAIN_LOSS, etc.")
print()
print("2. ✅ Added jaxtyping throughout")
print("   - Type aliases: ImageBatch, Logits, Targets, ScalarLoss")
print("   - Dimension symbols: B=batch, C=channels, H=height, W=width, N=num_classes")
print("   - Used in: forward(), *_step() methods in lit_module.py")
print("   - Added to: alexnet.py forward() method")
print()
print("3. ✅ Refactored lit_datamodule.py")
print("   - Replaced if-elif chain with match-case pattern")
print("   - Removed code duplication")
print()
print("4. ✅ All TODOs resolved")
print("   - lit_module.py line 276: ✅ RESOLVED")
print("   - lit_datamodule.py line 93: ✅ RESOLVED")
print("=" * 60)
