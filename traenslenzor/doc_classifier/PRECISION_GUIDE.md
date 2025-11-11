# Training Precision Configuration Guide

## Overview

The `TrainerFactoryConfig` provides two **independent** precision-related settings:

1. **`precision`** - PyTorch Lightning parameter controlling weight/activation **data types**
2. **`tf32_matmul_precision`** - PyTorch setting controlling how **FP32 operations execute** on Tensor Cores

These settings are **complementary, not mutually exclusive**. Understanding when to use each is key to optimal training performance.

---

## Precision Modes (`precision` parameter)

### FP32 (Full Precision) - `"32-true"`
```toml
[trainer_config]
precision = "32-true"
```

**What it does:**
- Model weights stored as 32-bit floats (FP32)
- All operations use FP32 arithmetic
- Maximum numerical stability and accuracy

**Use when:**
- First time training a model (baseline)
- Need maximum accuracy/reproducibility
- Debugging numerical issues
- Model is small and speed isn't critical

**Performance:** Slower than mixed precision, but works on all GPUs

---

### FP16 Mixed Precision - `"16-mixed"`
```toml
[trainer_config]
precision = "16-mixed"
```

**What it does:**
- Automatically converts operations to FP16 where safe
- Keeps model weights in FP32, creates FP16 copies for forward pass
- Uses loss scaling to prevent gradient underflow
- Powered by PyTorch's Automatic Mixed Precision (AMP)

**Use when:**
- Training large models (ResNet-50, ViT, etc.)
- Need maximum speed on modern GPUs (Pascal+)
- Have ample memory (FP16 uses less VRAM)

**Performance:** ~2x faster than FP32, ~50% less memory

**Gotchas:**
- Can cause numerical instability with some operations
- May require gradient scaling tuning
- Not all operations benefit (some stay FP32)

---

### BF16 Mixed Precision - `"bf16-mixed"`
```toml
[trainer_config]
precision = "bf16-mixed"
```

**What it does:**
- Like FP16 mixed precision, but uses BFloat16 (BF16) instead
- BF16 has same exponent range as FP32 → less prone to overflow/underflow
- Smaller mantissa (7 bits vs FP16's 10 bits) → slightly less precise

**Use when:**
- Training on Ampere+ GPUs (RTX 30xx/40xx, A100, H100)
- Need mixed precision without numerical instability
- Alternative to FP16 when encountering NaN/Inf issues

**Performance:** Similar to FP16 (~2x faster than FP32)

**Advantages over FP16:**
- No gradient scaling needed
- Better numerical stability
- Simpler to use

**Requirements:** Requires Ampere or newer GPU architecture

---

## TensorFloat-32 (`tf32_matmul_precision` parameter)

TF32 is a **hardware feature** on Ampere+ GPUs that accelerates **FP32 matrix multiplications** using Tensor Cores.

### How TF32 Works

| Mode | Description | Mantissa Bits | Speed | Accuracy |
|------|-------------|---------------|-------|----------|
| `"highest"` | Full IEEE FP32 (no TF32) | 23 bits | 1x (baseline) | 100% |
| `"high"` | TF32 enabled | 10 bits | ~2-3x | ~99.9% |
| `"medium"` | TF32 enabled (relaxed) | 10 bits | ~2-3x | ~99.8% |

**Key Point:** TF32 doesn't change data types—weights are still FP32. It only changes **how** matmuls execute on the GPU.

### Default: `"medium"` (Recommended)
```toml
[trainer_config]
precision = "32-true"
tf32_matmul_precision = "medium"
```

**What you get:**
- FP32 weights and activations
- 2-3x faster matmuls/convolutions
- Minimal accuracy loss (~0.1-0.2%)

**Best for:** Most training scenarios on RTX 30xx/40xx or A100/H100 GPUs

---

### Maximum Accuracy: `"highest"`
```toml
[trainer_config]
precision = "32-true"
tf32_matmul_precision = "highest"
```

**What you get:**
- Full IEEE FP32 precision (no approximation)
- Slower than TF32 (~2-3x slower)

**Use when:**
- Need bit-exact reproducibility
- Debugging numerical issues
- Comparing against reference implementations
- Training very sensitive models (GANs, etc.)

---

### Disable TF32: `None`
```toml
[trainer_config]
precision = "32-true"
tf32_matmul_precision = null  # or omit the field
```

**What you get:**
- Uses PyTorch's default behavior (usually allows TF32)

**Use when:**
- Want PyTorch to decide automatically
- Testing different configurations

---

## Recommended Combinations

### 1. Fast Training (Default) ⚡
```toml
[trainer_config]
precision = "32-true"
tf32_matmul_precision = "medium"
```

✅ **Best for most use cases**
- Good speed (~2-3x faster than pure FP32)
- Good accuracy (minimal loss)
- Works on RTX 30xx/40xx, A100, H100

---

### 2. Maximum Speed 🚀
```toml
[trainer_config]
precision = "bf16-mixed"  # or "16-mixed"
tf32_matmul_precision = "medium"  # Ignored—not using FP32!
```

✅ **Best for large models or limited GPU memory**
- Fastest possible training (~4-5x faster than pure FP32)
- Lowest memory usage
- BF16 preferred for better stability

**Note:** `tf32_matmul_precision` is ignored because you're not doing FP32 operations

---

### 3. Maximum Accuracy 🎯
```toml
[trainer_config]
precision = "32-true"
tf32_matmul_precision = "highest"
```

✅ **Best for reproducibility and debugging**
- Full IEEE FP32 precision
- Slower but most accurate
- Good for baseline runs

---

### 4. Balanced (Recommended for Ampere+) ⚖️
```toml
[trainer_config]
precision = "bf16-mixed"
tf32_matmul_precision = null  # Not needed
```

✅ **Best for modern GPUs (RTX 30xx+, A100/H100)**
- Fast training with BF16
- Good numerical stability
- Simple configuration

---

## Quick Decision Guide

```
Do you have an Ampere+ GPU (RTX 30xx/40xx, A100, H100)?
│
├─ YES → Is your model large (ResNet-50+, ViT, etc.)?
│        │
│        ├─ YES → Use precision="bf16-mixed", tf32_matmul_precision=null
│        │        (Fastest, stable, low memory)
│        │
│        └─ NO  → Use precision="32-true", tf32_matmul_precision="medium"
│                 (Good speed, good accuracy)
│
└─ NO  → Is speed critical?
         │
         ├─ YES → Use precision="16-mixed", tf32_matmul_precision=null
         │        (Fast mixed precision, older GPUs)
         │
         └─ NO  → Use precision="32-true", tf32_matmul_precision=null
                  (Standard FP32 training)
```

---

## Common Mistakes ❌

### ❌ Setting TF32 with Mixed Precision
```toml
precision = "16-mixed"
tf32_matmul_precision = "medium"  # ⚠️ Ignored! Wastes config space
```

**Why it's wrong:** Mixed precision uses FP16/BF16 natively, not FP32. The TF32 setting has no effect.

**Fix:** Set `tf32_matmul_precision = null` or omit it entirely when using mixed precision.

---

### ❌ Expecting TF32 to Work Like Mixed Precision
```toml
precision = "32-true"
tf32_matmul_precision = "medium"
# Expecting this to use 16-bit types ❌
```

**Why it's wrong:** TF32 is **not** the same as FP16/BF16. Your model still uses FP32 weights/activations. TF32 only affects how matmuls execute internally.

**Fix:** If you want 16-bit types, use `precision = "16-mixed"` or `"bf16-mixed"`.

---

### ❌ Using FP16 on Ampere+ GPUs
```toml
precision = "16-mixed"  # ⚠️ Suboptimal on Ampere+
```

**Why it's suboptimal:** Ampere+ GPUs support BF16, which is more stable and doesn't require gradient scaling.

**Fix:** Use `precision = "bf16-mixed"` on RTX 30xx/40xx, A100, H100.

---

## Validation Warnings

The `TrainerFactoryConfig` will warn you about suboptimal combinations:

```python
# This will trigger a warning:
config = TrainerFactoryConfig(
    precision="16-mixed",
    tf32_matmul_precision="medium"
)

# Warning: tf32_matmul_precision='medium' has no effect with precision='16-mixed'.
# TF32 only affects FP32 operations, but you're using mixed precision (FP16/BF16).
```

---

## Performance Benchmarks (Rough Estimates)

Based on training ResNet-50 on RTX 3080 Ti:

| Configuration | Speed | Memory | Accuracy |
|---------------|-------|--------|----------|
| `precision="32-true"`, `tf32_matmul_precision="highest"` | 1.0x | 100% | 100% (baseline) |
| `precision="32-true"`, `tf32_matmul_precision="medium"` | **2.5x** | 100% | 99.9% |
| `precision="16-mixed"` | **4.0x** | 60% | 99.5% |
| `precision="bf16-mixed"` | **4.0x** | 60% | 99.7% |

*Note: Actual speedups vary by model architecture and GPU.*

---

## References

- **PyTorch TF32 Docs:** https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
- **Lightning Precision Docs:** https://lightning.ai/docs/pytorch/stable/common/precision.html
- **NVIDIA Ampere Architecture:** https://www.nvidia.com/en-us/data-center/ampere-architecture/
- **BFloat16 Paper:** https://arxiv.org/abs/1905.12322

---

## Summary

- **`precision`** controls data types (FP32, FP16, BF16)
- **`tf32_matmul_precision`** controls FP32 matmul execution (only affects FP32!)
- **Don't mix them up:** TF32 ≠ mixed precision
- **Default is good:** `precision="32-true"` + `tf32_matmul_precision="medium"` for most cases
- **Use BF16 on Ampere+:** Better than FP16, no gradient scaling needed
