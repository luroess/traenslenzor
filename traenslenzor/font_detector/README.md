# Font Detector MCP Server

An MCP (Model Context Protocol) server for font detection and size estimation.

## Features

- **Font Name Detection**: Identify font names from images using HuggingFace's font-identifier model
- **Font Size Estimation**: Estimate font size in points using trained per-font MLP regressors

## Installation

### Prerequisites

```bash
# Install fonts
sudo pacman -S ttf-roboto ttf-roboto-mono inter-font ttf-lato ttf-ibm-plex

# Install Python dependencies using uv
uv sync

# Setup Git LFS (for model checkpoints and datasets)
git lfs install
```

### Setup

```bash
cd traenslenzor/font_detector

# Generate datasets and train models (takes ~5-10 minutes)
python train_all.py
```

This will:
1. Generate 10,000 training samples per font for 5 fonts
2. Train MLP models for each font
3. Save checkpoints to `checkpoints/` directory
4. Generate training report

## Usage

### As MCP Server

Run the server:

```bash
python -m traenslenzor.font_detector.server
```

The server exposes two tools:

#### 1. detect_font_name

Detect font name from an image.

**Input:**
```json
{
  "image_path": "/path/to/image.png"
}
```

**Output:**
```json
{
  "font_name": "Arial"
}
```

#### 2. estimate_font_size

Estimate font size from text box dimensions and content.

**Input:**
```json
{
  "text_box_size": [400, 64],
  "text": "Hello World",
  "font_name": "Roboto-Regular"
}
```

**Output:**
```json
{
  "font_size_pt": 12.5
}
```

### Command Line Tools

#### Generate Dataset

```bash
python -m traenslenzor.font_detector.font_size_model.data_gen \
  --font Roboto-Regular \
  --n-train 10000 \
  --n-val 1000 \
  --n-test 1000 \
  --output-dir data
```

#### Train Model

```bash
python -m traenslenzor.font_detector.font_size_model.train \
  --font Roboto-Regular \
  --data-dir data \
  --output-dir checkpoints \
  --epochs 100 \
  --batch-size 32
```

#### Inference

```bash
python -m traenslenzor.font_detector.font_size_model.infer \
  --font Roboto-Regular \
  --w 400 \
  --h 64 \
  --text "Hello World"
```

## Demo

Quick end-to-end demo:

```bash
# Install and train (first time only)
python train_all.py

# Test size estimation
python -m traenslenzor.font_detector.font_size_model.infer \
  --font Roboto-Regular \
  --w 400 \
  --h 64 \
  --text "Hello World"
```

## Architecture

### Font Name Detection
- Uses HuggingFace API: `gaborcselle/font-identifier`
- Automatic image preprocessing and resizing
- Returns top predicted font name

### Font Size Estimation

**Features (30-dimensional):**
- Width and height in pixels (2)
- Text length (1)
- Character density (1)
- Letter histogram a-z, normalized (26)

**Model:**
- Input: 30 features
- Hidden layer 1: 64 units + ReLU
- Hidden layer 2: 32 units + ReLU
- Output: 1 unit (font size in points)
- Loss: MSE
- Optimizer: Adam (lr=0.001)

**Training:**
- Per-font models (5 fonts total)
- 10,000 training samples per font
- Early stopping with patience=10
- Input normalization (standardization)

