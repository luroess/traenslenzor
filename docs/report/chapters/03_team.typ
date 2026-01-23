#import "@preview/supercharged-hm:0.1.2": *
#import "@preview/pintorita:0.1.4"
#show raw.where(lang: "pintora"): it => pintorita.render(it.text, style: "default")
#show figure: set block(breakable: true)


#let contributed(component, ..contributions) = [
  #heading(level: 3)[#component]
  #figure(caption: [Student contributions linked to #component])[
    #table(
      columns: (auto, auto, 1fr),
      align: (left, left, left),
      stroke: 0.5pt,
      inset: 6pt,
      table.header([*Student*], [*WP ID*], [*Contribution Details*]),
      ..contributions,
    )
  ]
]

= Team <team>

== Reflection <team_reflexion>

#strong[What went right:]

- Team members were largely able to work independently.
- Integration was mostly handled individually by team members, based on a shared understanding of what a session must contain.

#strong[What went wrong:]
- Some components were less related to machine learning and instead mainly involved traditional software engineering tasks.
- (Text Extractor) Unfortunately, PaddleOCR did not produce satisfactory results; this should have been evaluated more thoroughly before being integrated into the project.
- (Font Detector) The late decision to switch the OCR engine (from Paddle to Tesseract) caused regression in the Font Detector, as the model was trained on tight bounding boxes but received loose boxes from Tesseract.
- (Font Detector) The resolution mismatch between synthetic training data (72 DPI) and high-resolution images was initially overlooked because the primary test image also had 72 DPI. This caused the error to go unnoticed until late-stage testing with high-res images revealed massive size overestimation, which was addressed by implementing DPI scaling.
- #[
    (Supervisor) In retrospect, we should have evaluated more carefully which model to start with, as working with a suboptimal model consumed a considerable amount of time.
  ]

== Work Packages <team_work_packages>

#figure(caption: [Defined work packages])[
  #table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*WP ID*], [*Work Package Description*]),
    [UI1], [Basic UI],
    [UI2], [Streamlit UI revision: async handling, session init],
    [UI3], [Live updates (polling/progress) + session overview + image panel],
    [UI4], [Session tools and UI chores],

    [SV1], [Technology experiments.],
    [SV2], [Supervisor setup.],
    [SV3], [Mock infrastructure.],
    [SV4], [Evaluate different #gls("llm")s.],
    [SV5], [Multiple tool calls (LLaMA 3).],
    [SV6], [Memory.],
    [SV7], [Session changes.],
    [SV10], [Bug fixes.],

    [SV1], [Technology experiments.],
    [SV2], [Supervisor setup.],
    [SV3], [Mock infrastructure.],
    [SV4], [Evaluate different #gls("llm")s.],
    [SV5], [Multiple tool calls (LLaMA 3).],
    [SV6], [Memory.],
    [SV7], [Session changes.],
    [SV10], [Bug fixes.],

    [FS1], [File Server],
    [FS2], [Session Server],

    [DT1], [Initial testing with single-batch translation.],
    [DT2], [Tried different #gls("llm") models for translation quality.],
    [DT3], [Batch translation implementation with numbered output parsing.],

    [FT1], [Font detector roadmap, model review, and dataset planning.],
    [FT2], [Font detector MCP server, baseline MLP model, and dataset generation.],
    [FT3], [Dataset improvements, feature expansion, and MLP training results.],
    [FT4], [Custom ResNet classifier, cropping updates, and test integration.],

    [XDE1], [Direct Version],
    [XDE2], [Separate #gls("llm")],

    [IR1], [LaMa inpainting setup and testing],
    [IR2], [Text operations module],
    [IR3], [Rotation and transformation support],
    [IR4], [MCP server integration],

    // Document Scanner Jan Duchscherer
    [DS1], [Integrate UVDoc + runtime/backends],
    [DS2], [Deskew pipeline + SessionState integration],
    [DS3], [Super-resolution + stabilization],

    // Document Class Detector Jan Duchscherer
    [DC1], [Bootstrap ML infrastructure],
    [DC2], [RVL-CDIP dataset + Albumentations transforms + dataset stats],
    [DC3], [Lightning training pipeline (AlexNet/ResNet-50/ViT-B/16)],
    [DC4], [Experiment tooling (W&B, Optuna sweeps, resume, tests)],
    [DC5], [Interpretability (Captum attributions)],
    [DC6], [Serving tool (FastMCP + File Server session integration)],
  )
] <team_work_packages_work_packages_table>

#contributed(
  "UI",
  [Felix Schladt],
  [UI1],
  [
    - Set up the Streamlit #gls("ui").
    - Implemented the initial chat interface (prompt input, message rendering) and a simple run helper.
  ],
  [Jan Duchscherer],
  [UI2],
  [
    - Extended the UI into a session-based app with improved File Server integration.
    - Added prompt presets, refined chat-session handling & caching.
    - Updated session progress and image panel to match evolving supervisor interfaces.
  ],
  [Lukas Roess],
  [UI1],
  [
    - Improved run ergonomics (default Streamlit port).
    - Extended the sidebar session overview to surface additional per-item metadata (e.g., translation/font fields).
  ],
  [Felix Schladt],
  [UI2],
  [
    - Refactored progress tracking to rely on `SessionProgress` computed by the File Server.
    - Added caching/invalidation to keep the sidebar session overview consistent when sessions change.
  ],
  [Jan Duchscherer],
  [UI3],
  [
    - Implemented and refined live polling behavior (including the ability to disable polling) and environment configuration.
    - Ensured the session overview and image panel stay up-to-date while the supervisor runs.
  ],
  [Lukas Roess],
  [UI2],
  [
    - Improved UI compatibility with session state schema changes (nested translation/font structures).
    - Adapted the UI to changes in downstream tools (e.g., image renderer) to keep the pipeline functional end-to-end.
  ],
  [Jan Schaible],
  [UI2],
  [
    - Fixed rendering of the extracted document stage and improved periodic image refresh behavior.
  ],
  [Jan Duchscherer],
  [UI4],
  [
    - Added session export/import utilities and cleanup helpers.
    - Improved caching and polling.
  ],
)

#contributed(
  "Supervisor",
  [Felix Schladt\ Jan Schaible],
  [SV1],
  [
    - Compared LangChain vs. LangGraph.
    - Compared different methods for tool calling.
    - Investigated how information could be passed between tools.
  ],
  [Felix Schladt\ Jan Schaible],
  [SV2],
  [
    - Configured LangChain to use Ollama.
    - Used the Ollama #gls("api") to automatically pull models if not present.
    - Set up a #gls("repl") to interact with the #gls("llm").
    - Enabled tool calling in the #gls("llm").
  ],
  [Felix Schladt\ Jan Schaible],
  [SV3],
  [
    - Created a mock server for each tool.
    - Mocked each tool to return sample data.
    - Converted the app to async to work around the Python GIL @pep703.
  ],
  [Felix Schladt],
  [SV4],
  [
    // todo felix
    - Evaluated multiple #glspl("llm") available on hugging face as replacement for LLaMA 3.2.
    - Settled on qwen3 for its tool calling and comprehension abilities.
  ],
  [Jan Schaible],
  [SV5],
  [
    - Investigated why LLaMA 3 fails to call multiple tools back-to-back.
    - Extended Ollama with a custom model derived from LLaMA 3 and an updated template for back-to-back tool calls.
    - Discarded because LLaMA 3 was not reliable enough.
  ],
  [Felix Schladt\ Jan Schaible],
  [SV6],
  [
    - Created a tool the #gls("llm") can call to persist important information.
    - Injected that memory into the prompt so the #gls("llm") has it in-context.
    - Discarded because the #gls("llm") did not reliably store information relevant for later use.
  ],
  [Felix Schladt\ Jan Schaible],
  [SV7],
  [
    - Implemented changes based on the updated session concept.
    - Injected the session into the supervisor system prompt.
  ],
  [Felix Schladt\ Jan Schaible],
  [SV10],
  [
    - Optimized prompts.
    - Fixed minor bugs.
  ],
)

#contributed(
  "File Server",
  [Felix Schladt\ Jan Schaible],
  [FS1],
  [
    - Implemented an in-memory HTTP file server and client.
  ],
  [Felix Schladt\ Jan Schaible],
  [FS2],
  [
    - Implemented an in-memory HTTP session server and client.
  ],
)

#contributed(
  "Document Scanner",
  [Jan Duchscherer],
  [DS1],
  [
    - Integrated UVDoc and introduced a backend-based runtime.

  ],
  [Jan Duchscherer],
  [DS2],
  [
    - Improved document cropping from TE1 and added transition fallback logic.
    - Updated session state structures to carry scanner outputs for downstream tools.
  ],
  [Jan Duchscherer],
  [DS3],
  [
    - Iterated on flow-field based deskewing.
    - Added super-resolution tool & cut down interface complexity.
  ],
)

#contributed(
  "Text Extractor",
  [Felix Schladt],
  [TE1],
  [
    - Converted to grayscale, applied blurring, and detected edges with Canny
    - Document extraction and deskewing
  ],
  [Felix Schladt],
  [TE2],
  [
    - Performed #gls("ocr") using PaddleOCR.
    - Parsed results into the session data structure.
  ],
  [Jan Schaible],
  [TE3],
  [
    - Performed #gls("ocr") using Tesseract.
    - Parsed results into the session data structure.
  ],
)

#contributed(
  "Document Translator",
  [Benedikt Köhler\ Lukas Röß],
  [DT1],
  [
    - Validated translation quality on single-batch inputs.
    - Confirmed numbering preservation is required for correct mapping.
  ],
  [Benedikt Köhler\ Lukas Röß],
  [DT2],
  [
    - Compared different #gls("llm") models for translation quality.
    - Selected the most reliable model for the batch workflow.
  ],
  [Benedikt Köhler\ Lukas Röß],
  [DT3],
  [
    - Implemented a batch translation prompt that preserves line numbering.
    - Implemented robust regex-based response parsing with individual retry logic for text items missing from the batch response.
    - Added a sequential fallback mechanism for complete batch failures to ensure reliability.
  ],
)

#contributed(
  "Document Class Detector",
  [Jan Duchscherer],
  [DC1],
  [
    - Bootstrapped the `doc_classifier` subpackage and Config-as-Factory infrastructure (`BaseConfig`, `PathConfig`, `Console`).
    - Implemented TOML-driven configuration with typed CLI overrides.
    - Initial training and tuning runs.
  ],
  [Jan Duchscherer],
  [DC2],
  [
    - Integrated RVL-CDIP via Hugging Face Datasets and implemented model-aware Albumentations transform presets.
    - Added dataset statistics computation and deterministic subsampling for debugging.
    - Added support for LitTuner integration.
  ],
  [Jan Duchscherer],
  [DC3],
  [
    - Implemented Lightning training modules with modular backbones (AlexNet, ResNet-50, ViT-B/16).
    - Added custom fine-tune callbacks with OneCycleLR support.
    - Improved confusion-matrix logging via W&B.
    - MCP integration prep (model saving, loading, inference).
  ],
  [Jan Duchscherer],
  [DC4],
  [
    - Improved experiment tracking (W&B); allow easily querying past runs.
    - Add augmentation scheduling callback.
    - Revised Optuna integration for hyperparameter sweeps + pruning callbacks.
    - Optuna Sweeps and final training runs.
    - Improved config and CLI handling (resume runs, load checkpoints, test runs).
  ],
  [Jan Duchscherer],
  [DC5],
  [
    - Added Captum attribution utilities (Integrated Gradients, Grad-CAM, ...) for qualitative interpretability.
    - Run tests and attributions on tests split.
  ],
)

#contributed(
  "Font Detector",
  [Lukas Röß],
  [FT1],
  [
    - Created the roadmap for the font detector.
    - Reviewed font identifier models.
    - Planned synthetic dataset generation for five fonts and multiple sizes.
  ],
  [Lukas Röß],
  [FT2],
  [
    - Implemented MCP tools for font name detection and size estimation.
    - Integrated the HuggingFace font-identifier model.
    - Built dataset generation with 5 fonts and 10k samples per font.
    - Defined a 30D MLP with ReLU, MSE loss, Adam, and per-font normalization.
  ],
  [Lukas Röß],
  [FT3],
  [
    - Improved synthetic data with tight cropping and multiline text.
    - Expanded the feature vector to include line count and log features.
    - Trained per-font MLPs and recorded MAE and RMSE results.
  ],
  [Lukas Röß],
  [FT4],
  [
    - Replaced the HuggingFace model with a custom ResNet18.
    - Generated a dedicated synthetic dataset for the classifier.
    - Added testing scripts and debugged MCP integration.
    - Disabled the line-count feature due to unreliable behavior on real data.
    - Implemented DPI scaling to handle Tesseract's loose bounding boxes.
  ],
)

#contributed(
  "Document Editor",
  [Jan Schaible],
  [XDE1],
  [
    - Implemented a tool for the supervisor to modify session text.
    - Implemented a tool for the supervisor to retrieve the current session text.
  ],
  [Felix Schladt\ Jan Schaible],
  [XDE2],
  [
    - Implemented a tool to optimize the text stored in the session according to user instructions.
    - Implemented parsing.
    - Implemented an additional #gls("llm") call.
    - Implemented an #gls("mcp") server for the tool.
  ],
)

#contributed(
  "Document Image Renderer",
  [Benedikt Köhler],
  [IR1],
  [
    - Integrated the LaMa inpainting model with auto-download.
    - Evaluated PyTorch JIT and ONNX deployment strategies.
    - Implemented lazy model initialization for efficient resource usage.
    - Added unit tests and snapshot references for both backends.
  ],
  [Benedikt Köhler],
  [IR2],
  [
    - Extracted mask creation and text drawing into a dedicated module.
    - Implemented font handling with fallback support.
  ],
  [Benedikt Köhler],
  [IR3],
  [
    - Added rotation angle calculation from bounding box coordinates.
    - Implemented perspective transformation using OpenCV.
    - Created compositing logic to paste results onto originals.
  ],
  [Benedikt Köhler],
  [IR4],
  [
    - Built the FastMCP server exposing the replace_text tool.
    - Added session validation and structured error handling.
  ],
)

#pagebreak()

== Timeline <team_timeline>

#figure(caption: [Project timeline by work package])[
  ```pintora
    gantt
      dateFormat YYYY-MM-DDTHH
      axisFormat DD/MM
      axisInterval 2w

  section user_interface
  "Basic UI"       : 2026-1-10, 1w
  "Streamlit rev."       : ui2, 2025-12-14, 2w
  "Live updates (polling/progress) + session overview + image panel"       : ui2, 2025-12-22, 4w
  "Session tools and UI maintenance"       : ui3, 2026-1-20, 3d

  section supervisor
  "Technology experiments"       : sv1, 2025-10-13, 2w
  "Supervisor setup"       : sv2, after sv1, 2w
  "Mock Infrastructure"       : sv3, after sv2, 2w
  "Evaluate different LLMs"       : sv4, after sv3, 2w
  "Multiple tool calls (LLaMA 3)"       : sv5, after sv3, 2w
  "Memory"       : sv6, after sv5, 1w
  "Session changes"       : sv7, after sv6, 1w
  "Bug fixes"       : sv10, 2025-12-20, 2w

  section file_server
  "File Server"       : fs1, after sv3, 1w
  "Session Server"       : fs2, after sv6, 1w

  section text_extractor
  "Flatten Image"       : te1, after sv6, 1w
  "Paddle OCR"       : te2, after te1, 1w
  "Tesseract"       : te3, 2026-1-1, 1w

  section document_scanner
  "Integrate UVDoc + flatten image"       : ds1, 2026-01-03, 5d
  "Deskew pipeline + MCP integration"       : ds2, 2026-01-03, 19d
  "Super-resolution + stabilization"       : ds3, 2026-01-18, 4d

  section font_detector
  "Roadmap & Review"       : ft1, 2025-10-13, 1w
  "MCP & MLP Baseline"     : ft2, after ft1, 6w
  "Dataset & MLP Tuning"   : ft3, after ft2, 2w
  "ResNet & DPI Fix"       : ft4, after ft3, 6w

  section document_translator
  "Single-batch testing"       : dt1, 2025-11-24, 3w
  "LLM model comparison"       : dt2, after dt1, 1w
  "Batch implementation"       : dt3, after dt2, 4w

  section x_document_editor
  "Direct Version"       : xde1, 2025-12-25, 1w
  "Separate LLM"       : xde2, after xde1, 2w

  section document_image_renderer
  "LaMa inpainting setup"       : ir1, 2025-10-25, 1w
  "Text operations module"       : ir2, after ir1, 2w
  "Rotation and transformation"       : ir3, 2025-12-2, 2w
  "MCP server integration"       : ir4, after ir3, 1w

  section document_class_detector
  "Bootstrap ML infrastructure"       : dc1, 2025-10-13, 2w
  "RVL-CDIP dataset + initial transforms"       : dc2, 2025-10-24, 5d
  "Fixes: training pipeline (optimizer, callbacks)"       : dc3, 2025-11-13, 2w
  "MCP integration and serving"       : dc3, 2025-12-18, 35d
  "Improved augmentation + dataset stats":      : dc2, 2025-12-20, 3d
  "Optuna sweeps"       : dc4, 2025-12-20, 12d
  "Interpretability"       : dc5, 2026-01-16, 5d


  section Release
  "Release" : milestone, 2026-1-23, 0d
  ```
] <team_timeline_gantt>

