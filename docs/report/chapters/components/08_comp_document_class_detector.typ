#import "@preview/supercharged-hm:0.1.2": *
#import "@preview/wrap-it:0.1.1": wrap-content

#let ext_link_blue = rgb("#1a5fb4")
#let blink(dest, body) = link(dest)[
  #set text(fill: ext_link_blue)
  #body
]

== Document Classifier (RVL-CDIP) <comp_doc_cls>

The Document Class Detector assigns a document class label to the current session document (e.g., *invoice*, *letter*, *presentation*).
It is implemented in #blink("https://github.com/luroess/traenslenzor/tree/master/traenslenzor/doc_classifier")[`traenslenzor.doc_classifier`] and exposed to the supervisor as an #gls("mcp") server with a single tool (`classify_document(session_id)`).
Within the end-to-end pipeline, classification is pure enrichment step as the resulting class probabilities are not needed by any downstream component.\
The classifier consumes the deskewed document image produced by the `DocScanner` (@comp_doc_scanner) and persists a mapping of the top-k predicted classes with their probabilities to the session state and furthermore returns the same information to the MCP client.

Input: `SessionState.extractedDocument.id (str)` (deskewed image file id). \
Output: `SessionState.class_probabilities (dict[str, float])` (top-k class probabilities, k=3).

=== Training Stack

We implement an end-to-end PyTorch Lightning training stack for the 16-class RVL-CDIP dataset. The implementation follows a "_Config-as-Factory_" pattern: all runtime objects (dataset, transforms, Lightning module, trainer, callbacks) are instantiated from validated Pydantic configs via `BaseConfig.setup_target()` (see @fig-doc-classifier-config-diagram). This enables fully _declarative_ experiments configured through TOML files any CLI overrides.

//  TODO: reduce redundancy between the the elaboration on data here and in @doc-cls-data-prep
- Data pipeline: RVL-CDIP is loaded from the Hugging Face Hub (#blink("https://huggingface.co/datasets/chainyo/rvl-cdip")[`chainyo/rvl-cdip`]) and wrapped in a Lightning DataModule (`DocDataModule`) with split handling (`Stage`), caching (`PathConfig().hf_cache`), and optional deterministic subsampling (`limit_num_samples`) for fast debugging and sweeps.
- Augmentations & Transforms: Albumentations pipelines apply layout-preserving resize+pad preprocessing and optional grayscale #sym.arrow.r RGB conversion for ImageNet-initialized backbones. Normalization can use RVL-CDIP statistics (grayscale) or ImageNet constants (RGB).
- Backbones: custom grayscale AlexNet (from scratch) and torchvision ResNet-50 / ViT-B/16 with a task-specific classification head. Training supports head-only fine-tuning, full-backbone training and scheduled unfreezing of the backbone.
- Optimization and training: AdamW with explicit parameter groups (head vs backbone) and OneCycleLR scheduling.
- Monitoring: accuracy and cross-entropy loss are tracked with TorchMetrics; at the end of each validation epoch, a row-normalized confusion matrix is logged to Weights & Biases.
- Interpretability: Captum-based attributions (e.g., Integrated Gradients, Grad-CAM) can be used to perform post-hoc analysis of trained models.
// TODO: imnprove the

// #figure(caption: [Train and evaluation metrics tracked by the document classifier.])[
//   #code()[```py
//   class Metric(StrEnum):
//       TRAIN_LOSS = "train/loss"
//       TRAIN_ACCURACY = "train/accuracy"

//       VAL_LOSS = "val/loss"
//       VAL_ACCURACY = "val/accuracy"
//       VAL_CONFUSION_MATRIX = "val/confusion_matrix"

//       TEST_LOSS = "test/loss"
//       TEST_ACCURACY = "test/accuracy"
//   ```]]

==== Data, preprocessing & augmentations <doc-cls-data-prep>

For grayscale normalization (in [0, 1]), we compute dataset statistics on the RVL-CDIP train split via `DocDataModule.compute_grayscale_mean_std()`:
$mu = 0.911966$ and $sigma = 0.241507$ (single channel). These are the defaults in `TransformConfig`.

RVL-CDIP consists of grayscale document page images where *layout* is often the strongest cue (header structure, margins, tables, signature blocks).
Consequently, our preprocessing aims to preserve global layout while making the input compatible with both scratch-trained and ImageNet-pretrained backbones.

//  TODO: include some example images here!

*Data loading (Hugging Face).* We load RVL-CDIP from the Hugging Face Hub (`chainyo/rvl-cdip`) using `load_dataset(...)` and cache it locally (see `RVLCDIPConfig` in #blink("https://github.com/luroess/traenslenzor/blob/master/traenslenzor/doc_classifier/data_handling/huggingface_rvl_cdip_ds.py")[`traenslenzor/doc_classifier/data_handling/huggingface_rvl_cdip_ds.py`]).
Transforms are attached through `HFDataset.set_transform(...)` with a pickleable batch wrapper (`_TransformApplier`) to enable Lightning DataLoader multiprocessing.
For fast iteration and Optuna sweeps, `DocDataModule` supports deterministic subsampling via `limit_num_samples` (shuffle #sym.arrow.r select prefix) in `traenslenzor/doc_classifier/lightning/lit_datamodule.py`.

*Layout-preserving resize+pad.* All pipelines start with:
- `LongestMaxSize(max_size=img_size)` to scale the longer side to a fixed bound, preserving aspect ratio.
- `PadIfNeeded(min_height=img_size, min_width=img_size)` to obtain a square canvas without cropping.

This avoids destroying layout cues that are common failure modes for center-crop style pipelines.

*Channel handling and normalization.* The dataset is grayscale, but torchvision backbones (ResNet/ViT) are typically ImageNet-pretrained on RGB.
Therefore, `TransformConfig` supports (a) replicating grayscale to 3 channels (`convert_to_rgb=true`) and (b) selecting the normalization statistics:
- `normalization_mode="dataset"` uses a single-channel mean/std estimated from the RVL-CDIP train split (computed by `DocDataModule.compute_grayscale_mean_std()`).
- `normalization_mode="imagenet"` uses ImageNet constants for better alignment with pretrained backbones.

#figure(
  caption: [Config-as-Factory wiring for the classifier training stack.],
)[
  #image("/graphics/doc-classifier-config.svg", width: 100%)
] <fig-doc-classifier-config-diagram>


*Augmentation pipelines.* We provide five transform presets as Config-as-Factory targets in #blink("https://github.com/luroess/traenslenzor/blob/master/traenslenzor/doc_classifier/data_handling/transforms.py")[`traenslenzor/doc_classifier/data_handling/transforms.py`]:
- `TrainTransformConfig`: “document-friendly” heavy augmentation for scratch training (mild rotation/scale/shift, perspective, scanner-like noise/blur, brightness/contrast, gamma).
- `TrainHeavyTransformConfig`: stronger regularization for continued training (affine + distortions, compression/downscale artifacts, CLAHE, coarse/grid dropout).
- `FineTuneTransformConfig`: light augmentation for pretrained models (resize/pad + mild photometric jitter).
- `FineTunePlusTransformConfig`: moderate fine-tuning augmentation (small rotation/perspective + mild noise/blur + photometric jitter).
- `ValTransformConfig`: deterministic preprocessing for evaluation (resize/pad + channel/normalization only).

Concretely, `TrainTransformConfig` keeps geometry close to the original page ($<=$5° rotation, $<=$5% scale, $<=$2% shift) while simulating capture noise (Gaussian noise, mild blur, brightness/contrast, gamma).
`TrainHeavyTransformConfig` increases geometric variability ($<=$7° rotation + affine/shear + distortion) and adds harder degradations (JPEG compression, downscale, CLAHE, and dropout occlusions) to steer the model towards learning a variety of robust features.

In addition to choosing a preset per experiment via TOML, the trainer can switch from a lighter to a heavier augmentation policy late in training (implemented as `TrainTransformSwitchCallback`), which we use as a simple augmentation schedule to improve robustness without slowing early optimization.


The callback block in @fig-doc-classifier-config-diagram is configured via `TrainerCallbacksConfig` and uses standard Lightning callbacks where possible (see the callback index at #blink("https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html")[lightning.ai/docs/.../callbacks]).

- #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint")[ModelCheckpoint]: Persists the best checkpoint (configurable monitor + mode) to enable reproducible evaluation and deployment.
- #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#lightning.pytorch.callbacks.LearningRateMonitor")[LearningRateMonitor]: Logs per-step learning rates (important for debugging OneCycleLR and backbone/head LR scaling).
- #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping")[EarlyStopping]: Optionally stops training when the monitored validation metric stops improving (disabled by default in our config).
- #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Timer.html#lightning.pytorch.callbacks.Timer")[Timer]: Optionally enforces a maximum wall-clock duration for runs (useful for budgeted sweeps and comparability).
- #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichModelSummary.html#lightning.pytorch.callbacks.RichModelSummary")[RichModelSummary]: Emits a concise model summary into logs for quick sanity checks of parameter counts and module structure.
- `CustomTQDMProgressBar` (based on #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.TQDMProgressBar.html#lightning.pytorch.callbacks.TQDMProgressBar")[TQDMProgressBar]): A thin wrapper that hides `v_num` in the progress output.
- `CustomRichProgressBar` (based on #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichProgressBar.html#lightning.pytorch.callbacks.RichProgressBar")[RichProgressBar]): Same idea as above, but for Rich-based progress rendering.
- `OneCycleBackboneFinetuning` (custom; based on #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html#lightning.pytorch.callbacks.BaseFinetuning")[BaseFinetuning]): Freezes the backbone for a head-only warmup and unfreezes at a configured epoch without changing optimizer param groups (scheduler-safe for OneCycleLR).
- `TrainTransformSwitchCallback` (custom; #blink("https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html#lightning.pytorch.callbacks.Callback")[Callback]): Switches from moderate to heavy Albumentations training transforms at a chosen epoch by re-attaching the dataset transform.
- `PyTorchLightningPruningCallback` (Optuna integration): Prunes underperforming trials early based on the configured monitor (`OptunaConfig.monitor`) to speed up sweeps.

=== Experiment Management (Results Snapshot)

#let summary = json("/analysis/wandb_runs_summary.json")
#let r = (x, digits: 3) => if x == none { "n/a" } else { str(calc.round(x, digits: digits)) }

We log training runs with Weights & Biases (offline-capable logging, resume, artifact restore) and optionally run Optuna sweeps with pruning callbacks. Table below summarizes validation accuracy for the subset of runs with runtime > 1 hour (n = #summary.n_total_runs), computed by `docs/report/analysis/wandb_runs_analysis.py`.

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  [Model], [n], [Mean], [CI low], [CI high],
  ..for row in summary.stats_by_model {
    (
      [#row.model],
      [#row.n],
      [#r(row.mean)],
      [#r(row.ci_low)],
      [#r(row.ci_high)],
    )
  },
)



=== W&B Run Analysis (Duration > 1h)

#let pct = x => str(calc.round(100 * x, digits: 1)) + "%"

This section provides a focused scientific report on the W&B experiment history for the document classifier.
We restrict to runs with runtime > 1 hour (n = #summary.n_total_runs) and use `val/accuracy` as the primary metric.
All plots and tables are generated by `docs/report/analysis/wandb_runs_analysis.py` and stored under `/analysis/` and `/imgs/wandb_analysis/`.
Notebook-style execution is available in `/analysis/wandb_runs_analysis.typ` (Callisto + Jupyter), which can replay the analysis pipeline and embed the same figures.

==== Research question and hypothesis

We ask: *Why do AlexNet runs trained "from scratch" appear to outperform pretrained ResNet-50 and ViT-B/16?*
Our working hypothesis is that the observed gap is driven by *training regime* (full-backbone vs head-only fine-tuning)
and *backbone learning-rate scale*, rather than architectural superiority.

==== Descriptive overview
#wrap-content(
  align: top + right,
  column-gutter: 22pt,
  columns: (2fr, 1.15fr),
  [#figure(
    caption: [Bootstrap mean accuracy with 95% CI by model.],
  )[
    #image("/imgs/wandb_analysis/accuracy_by_model_ci.png", width: 100%)
  ]],
  [
    The distributions show that AlexNet dominates the upper tail of validation accuracy, while ViT lags.
    However, configuration inspection reveals that AlexNet runs are the only ones with full-backbone training (train_head_only = false)
    and the largest backbone LR scale (~ 1.0). ViT and ResNet runs are predominantly head-only with small backbone LR scale (0.01-0.107),
    constraining feature adaptation to the RVL-CDIP domain.
  ],
)


==== Mechanistic evidence: training regime and backbone update

#figure(
  caption: [Accuracy vs backbone LR scale (log).],
)[
  #image("/imgs/wandb_analysis/accuracy_vs_backbone_lr_scale.png", width: 86%)
]

#if summary.head_only_vs_full != none [
  #let diff = summary.head_only_vs_full
  Full-backbone training outperforms head-only by #r(diff.diff) accuracy points (95% CI: #r(diff.ci_low) to #r(diff.ci_high)).
  Sample sizes: #diff.n_head_only head-only runs, #diff.n_full full-backbone runs.
]

The backbone LR scale is the strongest signal in the data (bootstrap r ~ 0.76, CI [0.41, 0.99]).
This aligns with the mechanism: full-backbone training allows feature adaptation to the document domain,
whereas head-only fine-tuning constrains the model to ImageNet-oriented features.
AlexNet appears "better" primarily because it is allowed to learn the task end-to-end.


==== Top runs (val accuracy)

We report the final epoch's train/val losses and accuracies (last logged to W&B) and the best validation loss for the top runs.

// TODO: don't read the CSV in typst; inspect it and read the values manually and hardcode them here
#let top_runs_raw = csv("/analysis/wandb_top_runs.csv", row-type: dictionary)
#let num = x => if x == "" or x == "nan" { none } else { float(x) }
#let top_runs = (
  top_runs_raw
    .filter(it => num(it.at("val/accuracy")) != none)
    .sorted(key: it => -num(it.at("val/accuracy")))
    .slice(0, 5)
)

#let test_metrics = (
  "alexnet#1": (acc: 0.9921187162399292, loss: 0.030138805508613586),
  "resnet50-finetune-rvlcdip": (acc: 0.9576217532157898, loss: 0.1416090726852417),
  "vitb16-finetune-rvlcdip": (acc: 0.6499301791191101, loss: 1.139289140701294),
)
#let test_for = name => if name in test_metrics { test_metrics.at(name) } else { (acc: none, loss: none) }
#let r_opt = x => if x == none { "-" } else { r(x) }
#let best_val_losses = (
  "alexnet#1": 0.030138805508613583,
  "alexnet#2": 0.1035796850919723,
  "resnet50-finetune-rvlcdip": 0.3469084799289703,
  "vitb16-finetune-rvlcdip": 1.2762616872787476,
)
#let best_val_for = row => {
  let name = row.at("Name")
  if name in best_val_losses { best_val_losses.at(name) } else { num(row.at("val/loss")) }
}
#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, left, right, right, right, right, right, right, right),
  [Run], [Backbone], [Epoch], [Train acc], [Val acc], [Train loss], [Val loss (best)], [Test acc], [Test loss],
  ..for row in top_runs {
    (
      [#row.at("Name")],
      [#row.at("backbone")],
      [#r(num(row.at("epoch")))],
      [#r(num(row.at("train/accuracy_epoch")))],
      [#r(num(row.at("val/accuracy")))],
      [#r(num(row.at("train/loss_epoch")))],
      [#r_opt(best_val_for(row))],
      [#r_opt(test_for(row.at("Name")).at("acc"))],
      [#r_opt(test_for(row.at("Name")).at("loss"))],
    )
  },
)
==== Training dynamics and OneCycleLR schedule

W&B plot exports are SVGs that rely on HTML embedding and do not render reliably in Typst.
Therefore, we regenerate the plots from exported CSV histories (`docs/report/analysis/doc_cls_plots.py`).

#figure(
  caption: [Training curves and scheduler (generated from W&B history exports for selected runs).],
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 16pt,
    row-gutter: 14pt,
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Validation accuracy*]],
      [#image("/imgs/doc-cls/val-acc.svg", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Validation loss*]],
      [#image("/imgs/doc-cls/val-loss.svg", width: 100%)],
    )],

    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Training loss*]],
      [#image("/imgs/doc-cls/train-loss.svg", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*OneCycleLR schedule*]],
      [#image("/imgs/doc-cls/one-cycle-lr.svg", width: 100%)],
    )],
  )
]

==== Per-class performance (confusion matrices)

#figure(
  caption: [Row-normalized confusion matrices for selected runs (one per backbone).],
)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 16pt,
    row-gutter: 14pt,
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*AlexNet (val, epoch 15)*]],
      [#image("/imgs/doc-cls/alexnet_confmat_val15.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*ResNet-50 (val, epoch 9)*]],
      [#image("/imgs/doc-cls/resnet_confmat_val9.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*ViT-B/16 (val, epoch 9)*]],
      [#image("/imgs/doc-cls/vitb_confmat_val9.png", width: 100%)],
    )],
  )
]

The AlexNet confusion matrix shows near-perfect per-class recall in this snapshot, while the pretrained backbones exhibit confusion among semantically close document types (e.g., *invoice* vs *letter*, and *memo* vs *news*).

==== Scientific interpretation: Why AlexNet "from scratch" wins here

1. *Training regime dominates architecture.*
  AlexNet runs are trained end-to-end with backbone LR scale ~ 1.0.
  ViT/ResNet runs are head-only (or nearly frozen), preventing meaningful adaptation to RVL-CDIP.
  The data strongly support this mechanism (full-backbone > head-only, and high LR scale correlates with accuracy).

2. *Domain shift penalizes frozen backbones.*
  RVL-CDIP images are grayscale document pages with high-contrast glyph structure.
  ImageNet-pretrained backbones optimized for natural RGB textures can be mismatched when frozen,
  whereas a scratch-trained model can learn document-specific features (strokes, margins, paper noise).

3. *Optimization budget and capacity interplay.*
  Larger models require either more training steps or more backbone adaptation to realize their capacity.
  Under limited epochs and head-only fine-tuning, ViT/ResNet underfit the domain,
  while AlexNet's smaller capacity is sufficient when trained end-to-end.

4. *Batch size and stability.*
  The best AlexNet runs use smaller batches (~ 44), consistent with the modest negative correlation between batch size and accuracy.
  This likely improves generalization and gradient noise beneficial for small-data fine-tuning.

==== Recommendations (actionable)

- *Unfreeze and scale the backbone for ViT/ResNet.*
  Use full training with backbone LR scale in [0.1, 1.0]. Consider staged unfreezing if stability is an issue.
- *Reduce batch size for fine-tuning.*
  Target batch sizes in the 32-44 range; use gradient accumulation if needed.
- *Match normalization to training regime.*
  For full training, dataset normalization (RVL-CDIP mean/std) is appropriate; for head-only, verify that ImageNet normalization does not dominate early features.
- *Equalize budgets before concluding architectural superiority.*
  Compare models under matched epochs/steps and augmentation pipelines; otherwise, training regime dominates the outcome.

==== Conclusion

The observed performance gap is best explained by training regime rather than architecture.
AlexNet appears superior because it is trained end-to-end while ViT/ResNet are constrained by head-only fine-tuning.
Allowing ViT/ResNet to adapt their backbones (with appropriate LR scale and batch size) is the most likely path to closing or reversing the gap.



=== Optuna Hyperparameter Sweeps

#let optuna = json("/analysis/optuna/optuna_summary.json")
#let optuna_stats = json("/analysis/optuna/optuna_stats.json")
#let r4 = x => str(calc.round(x, digits: 4))
#let r3 = x => str(calc.round(x, digits: 3))

We summarize two Optuna studies: `doc-classifier-alexnet-sweep` and `doc-classifier`.
All objectives are minimized; lower is better. Plots show COMPLETE trials (black circles) and PRUNED trials (red X) using the last recorded intermediate value; OTHER states are gray squares.
Non-finite objectives are omitted, so some plots remain sparse when many trials terminate early.
Parameter importance uses Optuna's estimator when available and falls back to a Spearman-correlation proxy otherwise.

==== Study overview

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: (left, left, right, right, right, right, right),
  [Study], [DB], [Trials], [Complete], [Pruned], [Failed], [Best value],
  ..for row in optuna {
    (
      [#row.study_name],
      [#row.db_name],
      [#row.n_trials],
      [#row.n_complete],
      [#row.n_pruned],
      [#row.n_fail],
      [#r3(row.best_value)],
    )
  },
)

// TODO: Improve the integration of the Optuna analysis figures witht the text. Currently they are just dumped here without any explanatin
==== ResNet sweep (doc-classifier): categorical effects (exploratory)

#let resnet_tests = optuna_stats.at("doc-classifier").at("categorical_tests")
#let norm_test = resnet_tests.filter(it => it.param == "normalization_mode").first()
#let transform_test = resnet_tests.filter(it => it.param == "transform_config").first()
#let unfreeze_test = resnet_tests.filter(it => it.param == "backbone_unfreeze_at_epoch").first()

#let norm_groups = norm_test.group_stats.map(it => str(it.group)).join(", ")
#let transform_groups = transform_test.group_stats.map(it => str(it.group)).join(", ")
#let unfreeze_groups = unfreeze_test.group_stats.map(it => str(it.group)).join(", ")

This sweep contains many PRUNED and non-finite trials, so we treat the following observations as exploratory.
Among finite COMPLETE trials, we observe `normalization_mode` ∈ {#norm_groups} and `transform_config` ∈ {#transform_groups};
other settings largely terminate early and cannot be compared meaningfully.
For `backbone_unfreeze_at_epoch` ∈ {#unfreeze_groups}, the objective differences are small relative to noise.

#figure(
  caption: [ResNet sweep (doc-classifier): objective landscape and diagnostics.],
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 18pt,
    row-gutter: 16pt,
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Normalization mode*]],
      [#image("/imgs/optuna_analysis/doc-classifier_norm_mode_box.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Transform preset*]],
      [#image("/imgs/optuna_analysis/doc-classifier_transform_box.png", width: 100%)],
    )],

    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Backbone unfreeze epoch*]],
      [#image("/imgs/optuna_analysis/doc-classifier_unfreeze_box.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Optimization history*]],
      [#image("/imgs/optuna_analysis/doc-classifier_history.png", width: 100%)],
    )],

    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Parameter importance*]],
      [#image("/imgs/optuna_analysis/doc-classifier_importance.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Unfreeze epoch (scatter)*]],
      [#image(
        "/imgs/optuna_analysis/doc-classifier_scatter_trainer_config.callbacks.backbone_unfreeze_at_epoch.png",
        width: 100%,
      )],
    )],
  )
]

// TODO: ensure to use wrap-content or grid to make the layout better - all these figures shouldn't be as larger and in a single column!
==== AlexNet sweep (doc-classifier-alexnet-sweep)

#let study_alexnet = optuna.filter(it => it.study_name == "doc-classifier-alexnet-sweep").first()

Best trial (#study_alexnet.best_trial_number) value = #r3(study_alexnet.best_value).
This sweep focuses on AlexNet-specific hyperparameters such as dropout and low max_lr values.
We plot the objective relationships using `sns.regplot` with `lowess` and bootstrap (n_boot = 10000).


#figure(
  caption: [alexnet-sweep: weight_decay vs objective (lowess).],
)[
  #image("/imgs/optuna_analysis/doc-classifier-alexnet-sweep_reg_weight_decay.png", width: 86%)
]

#figure(
  caption: [alexnet-sweep: max_lr vs objective (lowess).],
)[
  #image("/imgs/optuna_analysis/doc-classifier-alexnet-sweep_reg_max_lr.png", width: 86%)
]

#figure(
  caption: [doc-classifier-alexnet-sweep: optimization history.],
)[
  #image("/imgs/optuna_analysis/doc-classifier-alexnet-sweep_history.png", width: 86%)
]

#figure(
  caption: [doc-classifier-alexnet-sweep: param importance (Spearman proxy).],
)[
  #image("/imgs/optuna_analysis/doc-classifier-alexnet-sweep_importance.png", width: 86%)
]

==== Cross-study interpretation

- The ResNet sweep's best finite trials use dataset normalization and `finetune_plus` transforms, but the comparison vs ImageNet or `train` is not statistically testable because those settings appear only in non-finite trials.
- Backbone unfreeze epoch shows only small differences between epoch 1 and 2 in this sweep and is likely a second-order effect relative to normalization and augmentation choice.
- The AlexNet sweep suggests sensitivity to `weight_decay` and `max_lr`, with low max_lr values dominating the best trials.

// uv run -m traenslenzor.doc_classifier.run --config_path .configs/alexnet_scratch.toml --stage TEST

=== Attributability and Interpretability <doc-cls-attrib>

We qualitatively inspect the classifier's decision cues using Captum-based attributions @captum.
For each backbone, we scan the test split and select the most confident correct prediction ("best") and the most confident incorrect prediction ("worst").
We compute attributions for the predicted class (i.e., the explanation targets the model decision, not the ground truth).

We export multiple attribution methods (Integrated Gradients, Input x Gradient, Noise Tunnel, Occlusion, DeepLift, LayerGradXActivation),
but focus on Grad-CAM here because it yields an intuitive spatial explanation that matches the layout-driven nature of RVL-CDIP.

*Grad-CAM.* Grad-CAM builds a coarse importance map on the last convolutional features by weighting each feature map by the global-average pooled gradient of the target logit and then upsampling to input resolution @gradcam:
$ L^c = op("ReLU")(sum_k alpha_k^c A^k) $ with $ alpha_k^c = 1/(H W) sum_(i,j) partial y^c / partial A^k_(i,j) $.
Because these feature maps are low-resolution, the overlays appear blocky; this is expected and still informative at the layout level.

// Attribution artifacts were generated via:
// uv run python -m traenslenzor.doc_classifier.run --stage=TEST --attribution_run.enabled=true ...

#let attr_root = "/imgs/attributions"
#figure(
  caption: [Grad-CAM overlays for representative best/worst samples (target = predicted class).],
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 12pt,
    row-gutter: 10pt,

    // AlexNet (scratch)
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*AlexNet (best)* -- GT: *scientific publication*, Pred: *scientific publication* (p = 0.064).]],
      [#image(attr_root + "/alexnet/best/sample_1/grad_cam/blended_heatmap.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*AlexNet (worst)* -- GT: *advertisement*, Pred: *scientific publication* (p = 0.064).]],
      [#image(attr_root + "/alexnet/worst/sample_1/grad_cam/blended_heatmap.png", width: 100%)],
    )],

    // ResNet-50 (finetuned)
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*ResNet-50 (best)* -- GT: *handwritten*, Pred: *handwritten* (p = 0.103).]],
      [#image(attr_root + "/resnet50/best/sample_1/grad_cam/blended_heatmap.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*ResNet-50 (worst)* -- GT: *advertisement*, Pred: *scientific publication* (p = 0.115).]],
      [#image(attr_root + "/resnet50/worst/sample_1/grad_cam/blended_heatmap.png", width: 100%)],
    )],
  )
] <fig-doc-cls-attrib-gradcam>

*Interpretation.* In correct cases, the attribution follows class-defining layout cues: dense text blocks and page frame for *scientific publication* and strong stroke/line structure for *handwritten*.
In the failure cases, both backbones confuse *advertisement* with *scientific publication*; Grad-CAM highlights high-contrast, title-like typography and centered foreground structure, suggesting a shortcut based on global layout and scan artifacts.

*Practical takeaway.* These examples support a "layout-first" strategy: most saliency concentrates on document structure rather than semantics.
Robustness could be improved by augmentations that decorrelate labels from borders/background (e.g., border randomization/cropping, stronger background perturbations, explicit margin masking).

=== Serving and Session Integration

For inference inside trÄnslenzor, the classifier is deployed as a lightweight FastMCP server. `DocClassifierRuntime` loads a Lightning checkpoint (if configured), applies the validation preprocessing (`ValTransformConfig`), and returns top-k predictions; if no checkpoint is present, a deterministic mock mode returns stable pseudo-probabilities for integration testing.

The MCP tool keeps its signature minimal (`session_id` only) by resolving all inputs/outputs through the File Server session model:

- Fetch `SessionState` and read `SessionState.extractedDocument.id`.
- Download the deskewed image via `FileClient`.
- Run inference and persist `SessionState.class_probabilities` via `SessionClient.update(...)`.

#figure(
  caption: [MCP inference sequence for document classification.],
)[
  #image("/graphics/doc-classifier-serving.svg", width: 86%)
] <fig-doc-classifier-serving-diagram>

=== Summary

- End-to-end Lightning training stack with modular backbones (AlexNet, ResNet-50, ViT-B/16) and model-aware preprocessing.
- Config-as-Factory composition with TOML export/import and typed CLI overrides.
- Experiment tooling: W&B logging, Optuna sweeps, optional Lightning tuning, and Captum interpretability hooks.
- Serving path as FastMCP tool integrated into the session-based File Server architecture.
