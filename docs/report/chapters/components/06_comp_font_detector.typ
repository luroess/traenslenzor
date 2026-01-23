#import "@preview/supercharged-hm:0.1.2": *

== Font Detector <comp_font_detector>

The Font Detector estimates the font name and font size for each text block.
It runs as an #gls("mcp") tool and uses two parts: `font_detection` for font names and `font_size` for size regression.

=== Requirements and Interfaces
- Input: text box image, text box size, text
- Output: font name, font size in points
- MCP tools:
  - `detect_font_name(image_path)` returns a font label
  - `estimate_font_size(text_box_size, text, image_path?, font_name?)` returns size and resolved font name
  - `detect_font(session_id)` updates all OCR items in a session

=== Font Name Detection
The font name detector uses a custom ResNet18 classifier trained on synthetic text images.
Training data uses 2,000 samples per font across five fonts and random text at sizes from 10 to 40 pt.
The input pipeline pads or center-crops to 224x224 and applies ImageNet normalization.
During document runs, the MCP tool crops a dense 224x224 region based on OCR boxes.
The detector returns the top class label.

=== Font Size Estimation
Font size estimation is a per-font regression task.
The MLP is implemented from scratch in NumPy in a tiny_diff style with manual gradients.
The model uses ReLU and MSE loss and trains with Adam and early stopping.
Train, val, and test splits are generated with a synthetic text box generator.
The MLP uses 64 and 32 hidden units and outputs one size value.

The feature vector is 34-dimensional:
- width and height in pixels
- text length and character density
- log width, log height, log length, log density
- 26-bin letter histogram over a-z

Inputs are normalized with a per-font mean and standard deviation.
The estimator loads the correct model by font name and falls back to Roboto-Regular if a model is missing.

=== MCP Integration
The server lives in `traenslenzor/font_detector/mcp.py`.
The `detect_font` tool downloads the deskewed image from the File Server.
It runs global font detection and estimates size for each text item from its bounding box and text.

=== Training Results

==== Font Name Classification (ResNet18)
The classification model (ResNet18) was trained for 5 epochs on 10,000 synthetic images (2,000 per class).
The model achieved high accuracy on the specific synthetic validation set, demonstrating the ability to distinguish the five target fonts under ideal conditions.

#figure(
    table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: center,
    [*Metric*], [*Value*], [*Set*],
    [Accuracy], [99.2%], [Validation],
    [Precision], [0.99], [Validation],
    [Recall], [0.99], [Validation],
  ),
  caption: [Font name classification performance on synthetic validation set],
)

==== Font Size Estimation (MLP)
For font size estimation, separate Multi-Layer Perceptrons (MLPs) were trained for each of the five fonts. Each model was trained on 10,000 synthetic text box samples using a 34-dimensional feature vector (containing dimensions, text length, and log-transformed statistics). The models converged effectively, achieving an average Mean Absolute Error (MAE) of less than 0.9 points across all fonts on the synthetic test set.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    [*Font*], [*MAE*], [*RMSE*], [*Epoch*],
    [Roboto-Regular], [0.88 pt], [1.41 pt], [46],
    [RobotoMono-Regular], [0.87 pt], [1.32 pt], [74],
    [Inter-Regular], [0.81 pt], [1.38 pt], [46],
    [Lato-Regular], [0.81 pt], [1.33 pt], [68],
    [IBMPlexSans-Regular], [0.82 pt], [1.33 pt], [62],
    table.hline(),
    [*Average*], [*0.84 pt*], [*1.35 pt*], [-],
  ),
  caption: [Results are taken from `traenslenzor/font_detector/checkpoints/training_summary.json`.],
)

=== Final Evaluation <font_eval_final>

A final end-to-end run was performed using the `scientific_publication_Lato_` `straight` document to evaluate the system after the migration to Tesseract OCR.

==== Challenges with OCR Migration
The training data for the size regression model was generated using "tight" bounding boxes around the ink. Tesseract OCR, however, produces "loose" bounding boxes that include line spacing and ascender/descender room. This mismatch initially caused significant size overestimation.

To mitigate this without retraining, a geometric DPI calculation (based on standard A4 width) and a dynamic "Smart Crop" algorithm were implemented. The Smart Crop analyzes pixel density within the OCR box to measure the actual ink height before inference.

==== Metric Results
The following results compare the ground truth (`scientific_publication_metadata.json`) against the detecting values (`scientific_publication_session_details.csv`).

#figure(caption: [Font Size Detection accuracy on Scientific Publication Sample])[
  #table(
    columns: (auto, auto, auto, auto),
    inset: 5pt,
    align: center,
    [*Element*], [*Expected (pt)*], [*Detected (pt)*], [*Error (pt)*],
    [Abstract], [10], [12], [+2],
    [Section (Intro)], [12], [9], [-3],
    [Body Text], [10], [10-11], [+0 to +1],
    [Footer], [9], [10], [+1],
  )
]

#figure(
  image("../../imgs/scientific_publication_Lato_session_details.png", width: 50%),
  caption: [Session details view showing detected font properties],
) <fig:font_det_session>

==== Analysis
- *Size Accuracy:* The ink-measurement logic proved effective, bringing the body text error down to negligible levels (~1pt). Section headers were slightly underestimated, likely due to their sparse ink density relative to their bounding box height.
- *Font Identification:* The system correctly identified the font as *Lato-Regular* with *42.87% confidence*. While the confidence score is lower than ideal (likely due to the image quality and the dense scientific layout), the classification was correct.

#figure(
  image("../../imgs/scientific_publication_Lato_with_detection_box.png", width: 80%),
  caption: [Used scientific publication sample for final evaluation with detected bounding box],
) <fig:font_det_box>
