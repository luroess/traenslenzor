# Inpainter
## Inpainter Model
the current code uses the JIT version of https://github.com/advimman/lama this way we don't need to pull any code and the model is more portable.


## ONNX Inpainter Model
https://colab.research.google.com/github/Carve-Photos/lama/blob/main/export_LaMa_to_onnx.ipynb

with this notebook there is a way to convert the LaMa inpainter to ONNX format and change the input resolution.
i couldn't make this run on my macbooks gpu backend yet. so the jit version is much faster.
