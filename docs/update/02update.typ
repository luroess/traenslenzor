#import "template.typ": *

#show: slides.with(
  title: "Update Meeting - tränslenzor",
  date: datetime.today().display("[day]. [month repr:long] [year]"),
  ratio: 16/9,
  layout: "medium",
  title-color: rgb("#fb5555"),
)


== Supervisor

- Nutzereingab von Sprache & Dokument funktioniert
- Dateien werden auf einen Fileserver hochgeladen.
  - Dateien besitzen eine uuid über welche diese von anderen tools bezogen werden können.
  - Http basiert
- Prompt "engineered"

== Layout detector

- MCP server
- Einbinden von Paddle OCR 
  - Textbox extraction
  - Text OCR detection
- Möglichkeiten zur implementation eines eigenene Schrift-Locations Modells angeschaut.


== Document Image Renderer - What has been done
- implemented JIT version of LaMa (can use cpu, mps and probably cuda backends)
- tested ONNX version as well, but didn’t provide wished performance (CoreML issues, restricted size)
- added tests for it (snapshot testing with threshold, basic unit tests for utilities)
- read LaMa and Fast Fourier Convolution Paper → Better understanding of what LaMa actually does

== Document Image Renderer - What will be done
- add mcp server for the llm to actually use image renderer
- take transformation matrix into account
- use the mask to only add the inpainted and replaced text in the original image (mitigates small changes introduced by LaMa)
- integrate text rotation (currently only horizontal left to right text is supported)

== Document Image Renderer - Questions
- Best practises when model authors don't provide portable version?
- where is JIT vs ONNX vs using the actual code placed in production apps?


== Document Font Detector — Lukas Röß

*Tasks worked on*
- Implemented MCP server with font detection and size estimation tools
- Integrated HuggingFace font-identifier model for font name detection
- Created synthetic dataset generation (5 fonts, 10k samples each)
- Defined MLP architecture for font size regression:
  - 30D feature vector (width_px, height_px, text_len, character density, letter histogram)
  - Hidden layers: 64 → ReLU → 32 → ReLU → 1 output
  - MSE loss function with Adam optimizer
  - Per-font models with feature normalization

*Problems*
- HuggingFace font-identifier model has bad accuracy on tested target fonts

*Plans for Next Weeks*
- Train and evaluate MLP models on all fonts
- Optimize model hyperparameters
- Integrate trained models with MCP server


== Document Class Detector - What has been done?

- Implemented `RVL-CDIP` Dataset via _HuggingFace Datasets_
- Implemented hyperparameter tuning via _Optuna_ and _PyTorch Lightning's_ `Tuner`
- Revised modular augmentation pipelines with _Albumentations_
- Implemented confusion matrix logging and visualization
- Improved Callback system (BackboneFinetuning, EarlyStopping, ModelCheckpoint, LRMonitor, ...)
- Added CLI support for running experiments via _Pydantic_ configs
- Allow configuration via `.toml` files
- Improved logging integration with _WandB_
- Improved HParam Handling via Optuna
- Ran initial training experiments with all 3 baseline models (AlexNet, ResNet-50, ViT)

== Document Class Detector - Plans for Next Week
- Train baseline models (AlexNet, ResNet-50, ViT) on full RVL-CDIP dataset
- Provide MCP interface for supervisor integration
- Find optimal initial LR with LR Finder, and do further hyperparameter tuning using Optuna
- Understand and configure `BackboneFinetuning` schedule
- Captum interpretability analysis