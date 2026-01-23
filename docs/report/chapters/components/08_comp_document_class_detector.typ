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

#figure(
  caption: [Document-classifier test examples.],
)[
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    column-gutter: 10pt,
    row-gutter: 4pt,
    align: top,
    [
      #stack(
        spacing: 3pt,
        [#image("/imgs/doc-cls/advertisment_test.png", width: 95%)],
        [#text(size: 8pt)[(a) Advertisement.]],
      )
    ],
    [
      #stack(
        spacing: 3pt,
        [#image("/imgs/doc-cls/handwritten_test.png", width: 95%)],
        [#text(size: 8pt)[(b) Handwritten.]],
      )
    ],
    [
      #stack(
        spacing: 3pt,
        [#image("/imgs/doc-cls/file_folder_test.png", width: 95%)],
        [#text(size: 8pt)[(c) File folder.]],
      )
    ],
    [
      #stack(
        spacing: 3pt,
        [#image("/imgs/doc-cls/scientific_publication_test.png", width: 95%)],
        [#text(size: 8pt)[(d) Scientific publication.]],
      )
    ],
  )
] <fig-doc-cls-test-examples>

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

// The backbone LR scale is the strongest signal in the data (bootstrap r ~ 0.76, CI [0.41, 0.99]).
// This aligns with the mechanism: full-backbone training allows feature adaptation to the document domain,
// whereas head-only fine-tuning constrains the model to ImageNet-oriented features.
// AlexNet appears "better" primarily because it is allowed to learn the task end-to-end.


==== Training results and analysis <doc-cls-training-results>

We report the final epoch's train/val losses and accuracies (last logged to W&B) and the best validation loss for the top runs.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
  align: (left, left, right, right, right, right, right, right, right),
  stroke: 0.5pt,
  inset: 6pt,
  table.header(
    [*Run*],
    [*Backbone*],
    [*Epoch*],
    [*Train acc*],
    [*Val acc*],
    [*Train loss*],
    [*Val loss (best)*],
    [*Test acc*],
    [*Test loss*],
  ),
  [alexnet#1], [ALEXNET], [15], [0.954], [0.992], [0.144], [0.030], [0.992], [0.030],
  [alexnet#2], [ALEXNET], [7], [0.922], [0.968], [0.242], [0.104], [-], [-],
  [resnet50-finetune-rvlcdip], [RESNET50], [9], [0.976], [0.912], [0.078], [0.347], [0.958], [0.142],
  [resnet50-finetune-rvlcdip], [RESNET50], [9], [0.808], [0.819], [0.636], [0.347], [0.958], [0.142],
  [vitb16-finetune-rvlcdip], [VIT_B16], [9], [0.607], [0.622], [1.313], [1.276], [0.650], [1.139],
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
      [#image("/imgs/doc-cls/val-acc.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Validation loss*]],
      [#image("/imgs/doc-cls/val-loss.png", width: 100%)],
    )],

    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*Training loss*]],
      [#image("/imgs/doc-cls/train-loss.png", width: 100%)],
    )],
    [#stack(
      spacing: 4pt,
      [#text(size: 9pt)[*OneCycleLR schedule*]],
      [#image("/imgs/doc-cls/one-cycle-lr.png", width: 100%)],
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
#let r3 = x => str(calc.round(x, digits: 3))

// Formatting helper for small hyperparameters (e.g., max_lr, weight_decay).
#let fmt = x => {
  if x == none { "-" } else {
    let ax = calc.abs(x)
    let digits = if ax < 0.001 { 7 } else if ax < 0.01 { 6 } else { 4 }
    str(calc.round(x, digits: digits))
  }
}

We tune a small number of training hyperparameters with Optuna to reduce manual trial-and-error.
We summarize two studies: `doc-classifier` (ResNet-50 fine-tuning) and `doc-classifier-alexnet-sweep` (AlexNet from scratch).
The objective is the validation loss (cross-entropy); all studies are minimized (lower is better).

In the plots below, COMPLETE trials are shown as black circles and PRUNED trials as red X (using the last reported intermediate value).
Some figures are sparse because non-finite objectives are omitted.

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

==== ResNet-50 fine-tuning sweep (doc-classifier)

#let study_resnet = optuna.filter(it => it.study_name == "doc-classifier").first()
#let p_resnet = study_resnet.best_params

This sweep targets the ResNet-50 fine-tune regime and focuses on learning-rate schedule and regularization.

*Search space (from the Optuna database).* We sample:
- `datamodule_config.train_ds.transform_config`: categorical {`train`, `finetune`, `finetune_plus`}
- `datamodule_config.train_ds.transform_config.normalization_mode`: categorical {`dataset`, `imagenet`}
- `module_config.scheduler.max_lr`: log-uniform [1e-4, 3e-3]
- `module_config.scheduler.pct_start`: uniform [0.05, 0.25]
- `module_config.optimizer.backbone_lr_scale`: log-uniform [0.01, 0.2]
- `module_config.optimizer.weight_decay`: log-uniform [1e-6, 1e-2]
- `trainer_config.callbacks.backbone_unfreeze_at_epoch`: integer {1, 2, 3}

*Best trial.* Trial #study_resnet.best_trial_number achieves objective #r3(study_resnet.best_value) with:
- `transform_config`: #p_resnet.at("datamodule_config.train_ds.transform_config")
- `max_lr`: #fmt(p_resnet.at("module_config.scheduler.max_lr"))
- `pct_start`: #fmt(p_resnet.at("module_config.scheduler.pct_start"))
- `backbone_lr_scale`: #fmt(p_resnet.at("module_config.optimizer.backbone_lr_scale"))
- `weight_decay`: #fmt(p_resnet.at("module_config.optimizer.weight_decay"))
- `backbone_unfreeze_at_epoch`: #p_resnet.at("trainer_config.callbacks.backbone_unfreeze_at_epoch")

*Notes (interpretation).* 
- The best configuration uses `finetune_plus` with early backbone adaptation (`backbone_unfreeze_at_epoch = 2`), consistent with the hypothesis that RVL-CDIP benefits from stronger domain-specific augmentation and feature adaptation.
- With OneCycleLR, the effective backbone peak learning rate is `max_lr * backbone_lr_scale`, here approximately #fmt(p_resnet.at("module_config.scheduler.max_lr") * p_resnet.at("module_config.optimizer.backbone_lr_scale")). This keeps backbone updates conservative while allowing the head to train at the full `max_lr`.
- The top trials are tightly clustered (best value #r3(study_resnet.best_value)), suggesting a relatively flat optimum under the fixed trial budget; remaining variance is likely dominated by stochasticity (data order, augmentation randomness).

#wrap-content(
  align: top + right,
  columns: (1fr, 18em),
  column-gutter: 24pt,
  [
    #figure(
      caption: [ResNet sweep diagnostics (objective history and param importance).],
    )[
      #grid(
        columns: 1fr,
        row-gutter: 10pt,
        [#image("/imgs/optuna_analysis/doc-classifier_history.png", width: 100%)],
        [#image("/imgs/optuna_analysis/doc-classifier_importance.png", width: 100%)],
      )
    ] <fig-doc-cls-optuna-resnet-core>
  ],
  [
    The optimization history shows fast convergence among the few finite COMPLETE trials.
    The importance proxy indicates that the OneCycleLR peak learning rate (`max_lr`) and schedule shape (`pct_start`) dominate this sweep, with regularization and unfreezing acting as secondary knobs.
  ],
)

==== ResNet sweep: categorical effects (exploratory)

#let resnet_tests = optuna_stats.at("doc-classifier").at("categorical_tests")
#let norm_test = resnet_tests.filter(it => it.param == "normalization_mode").first()
#let transform_test = resnet_tests.filter(it => it.param == "transform_config").first()
#let unfreeze_test = resnet_tests.filter(it => it.param == "backbone_unfreeze_at_epoch").first()

#let norm_groups = norm_test.group_stats.map(it => str(it.group)).join(", ")
#let transform_groups = transform_test.group_stats.map(it => str(it.group)).join(", ")
#let unfreeze_groups = unfreeze_test.group_stats.map(it => str(it.group)).join(", ")

This sweep contains many PRUNED and non-finite trials, so we treat the following observations as exploratory.
Among finite COMPLETE trials, we observe `normalization_mode` in {#norm_groups} and `transform_config` in {#transform_groups}.
For `backbone_unfreeze_at_epoch` in {#unfreeze_groups}, differences are small relative to noise.

#figure(
  caption: [ResNet sweep (doc-classifier): categorical effects and unfreeze diagnostics.],
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 16pt,
    row-gutter: 14pt,
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
      [#text(size: 9pt)[*Unfreeze epoch (scatter)*]],
      [#image(
        "/imgs/optuna_analysis/doc-classifier_scatter_trainer_config.callbacks.backbone_unfreeze_at_epoch.png",
        width: 100%,
      )],
    )],
  )
] <fig-doc-cls-optuna-resnet-cats>

*Notes.* The box plots and group statistics suggest that `finetune_plus` is the only transform preset that yields a sizable set of finite COMPLETE trials (n = #transform_test.group_stats.first().n) with low spread (median #r3(transform_test.group_stats.first().median), std #r3(transform_test.group_stats.first().std)).
For backbone unfreezing, `backbone_unfreeze_at_epoch = 2` has a slightly better median objective (#r3(unfreeze_test.group_stats.at(1).median)) than epoch 1 (#r3(unfreeze_test.group_stats.at(0).median)), but the epoch-1 group has very small n and the ANOVA test is not significant.

==== AlexNet sweep (doc-classifier-alexnet-sweep)

#let study_alexnet = optuna.filter(it => it.study_name == "doc-classifier-alexnet-sweep").first()
#let p_alexnet = study_alexnet.best_params

This study is tailored to AlexNet-from-scratch training and focuses on dropout, weight decay, and a small `max_lr` range.

*Search space (from the Optuna database).* We sample:
- `module_config.model_params.dropout_prob`: uniform [0.2, 0.6]
- `module_config.optimizer.weight_decay`: log-uniform [2e-4, 1.2e-3]
- `module_config.scheduler.max_lr`: log-uniform [8e-5, 4e-4]

*Best trial.* Trial #study_alexnet.best_trial_number achieves objective #r3(study_alexnet.best_value) with:
- `dropout_prob`: #fmt(p_alexnet.at("module_config.model_params.dropout_prob"))
- `weight_decay`: #fmt(p_alexnet.at("module_config.optimizer.weight_decay"))
- `max_lr`: #fmt(p_alexnet.at("module_config.scheduler.max_lr"))

*Notes (interpretation).*
- The sweep is prune-heavy (trials: #study_alexnet.n_trials, complete: #study_alexnet.n_complete, pruned: #study_alexnet.n_pruned), indicating that many configurations diverge or underperform early under the fixed epoch budget.
- The importance proxy ranks `max_lr` as dominant (consistent with OneCycleLR sensitivity), followed by `weight_decay` and then dropout; this aligns with the typical failure mode of too aggressive learning rates causing unstable optimization.
- The best parameters sit near the lower end for dropout and in a narrow learning-rate regime, suggesting that capacity-limited AlexNet benefits more from stable optimization than from strong regularization.

#wrap-content(
  align: top + right,
  columns: (1fr, 18em),
  column-gutter: 24pt,
  [
    #figure(
      caption: [AlexNet sweep diagnostics (objective history and param importance).],
    )[
      #grid(
        columns: 1fr,
        row-gutter: 10pt,
        [#image("/imgs/optuna_analysis/doc-classifier-alexnet-sweep_history.png", width: 100%)],
        [#image("/imgs/optuna_analysis/doc-classifier-alexnet-sweep_importance.png", width: 100%)],
      )
    ] <fig-doc-cls-optuna-alexnet-core>
  ],
  [
    The importance plot ranks `max_lr` as the dominant factor, followed by `weight_decay` and then dropout.
    This matches the LOWESS trends in @fig-doc-cls-optuna-alexnet-reg, where only a narrow range of learning rates yields stable training.
  ],
)

We visualize objective relationships with `sns.regplot` (LOWESS smoothing, bootstrap n_boot = 10000).
#figure(
  caption: [AlexNet sweep: objective vs key hyperparameters (LOWESS).],
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 14pt,
    row-gutter: 12pt,
    [#image("/imgs/optuna_analysis/doc-classifier-alexnet-sweep_reg_weight_decay.png", width: 100%)],
    [#image("/imgs/optuna_analysis/doc-classifier-alexnet-sweep_reg_max_lr.png", width: 100%)],
  )
] <fig-doc-cls-optuna-alexnet-reg>

*Notes.* The LOWESS curves visually support a narrow "safe" region for `max_lr` (too large values correlate with higher objectives / pruning) and a milder effect for `weight_decay`. Because these are observational relationships (and include pruned trials), we treat them as guidance for a follow-up ablation rather than a causal conclusion.

==== Cross-study interpretation

- The ResNet sweep's best finite trials use dataset normalization and `finetune_plus` transforms; comparisons against ImageNet normalization remain inconclusive because those settings rarely produce finite COMPLETE trials.
- Backbone unfreezing shows only small differences and appears second-order relative to augmentation/normalization and learning-rate schedule.
- The AlexNet sweep shows the strongest sensitivity to `max_lr` and `weight_decay`, consistent with the importance plot and the LOWESS trends.

// uv run -m traenslenzor.doc_classifier.run --config_path .configs/alexnet_scratch.toml --stage TEST

=== Attributability and Interpretability <doc-cls-attrib>

We qualitatively inspect the classifier's decision cues using Captum-based attributions @captum.
For each backbone, we scan the test split and select the most confident correct prediction ("best") and the most confident incorrect prediction ("worst").
We compute attributions for the predicted class (i.e., the explanation targets the model decision, not the ground truth).

We export multiple attribution methods (Integrated Gradients, Input x Gradient, Noise Tunnel, Occlusion, DeepLift, LayerGradXActivation),
focus on Grad-CAM here because it yields an intuitive spatial explanation that matches the layout-driven nature of RVL-CDIP.
As a complementary view, we also show an Integrated Gradients (IG) map for a representative AlexNet sample.

*Grad-CAM.* Grad-CAM builds a coarse importance map on the last convolutional features by weighting each feature map by the global-average pooled gradient of the target logit and then upsampling to input resolution @gradcam:
$ L^c = op("ReLU")(sum_k alpha_k^c A^k) $ with $ alpha_k^c = 1/(H W) sum_(i,j) partial y^c / partial A^k_(i,j) $.
Because these feature maps are low-resolution, the overlays appear blocky; this is expected and still informative at the layout level.

// Attribution artifacts were generated via:
// uv run python -m traenslenzor.doc_classifier.run --stage=TEST --attribution_run.enabled=true ...

#let attr_root = "/imgs/attributions"
#figure(
  caption: [Attribution heatmaps for representative best/worst samples (target = predicted class).],
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 12pt,
    row-gutter: 10pt,

    // AlexNet (scratch)
    [#figure(
      image(attr_root + "/alexnet/ig_heat_best_scientific_alexnet_heatmap.png", width: 100%),
      caption: [IG heatmap — AlexNet (best) — GT+Pred: *scientific publication* (p = 0.063).],
    ) <fig-doc-cls-attrib-gradcam-alexnet-best>],
    [#figure(
      image(attr_root + "/alexnet/worst/sample_1/grad_cam/blended_heatmap.png", width: 100%),
      caption: [AlexNet (worst) — GT: *advertisement*, Pred: *scientific publication* (p = 0.064).],
    ) <fig-doc-cls-attrib-gradcam-alexnet-worst>],

    // ResNet-50 (finetuned)
    [#figure(
      image(attr_root + "/resnet50/blended_heatmap_best_handwritten_0.1_layer_grad_x_activ.png", width: 100%),
      caption: [ResNet-50 (best) — GT+Pred: *handwritten* (p = 0.100).],
    ) <fig-doc-cls-attrib-gradcam-resnet-best>],
    [#figure(
      image(attr_root + "/resnet50/worst/sample_1/grad_cam/blended_heatmap.png", width: 100%),
      caption: [ResNet-50 (worst) — GT: *advertisement*, Pred: *scientific publication* (p = 0.115).],
    ) <fig-doc-cls-attrib-gradcam-resnet-worst>],
  )
] <fig-doc-cls-attrib-gradcam>


resnet-worst uses Gradient-weighted Class Activation Mapping (Grad-CAM) @gradcam for attribution,which builds a coarse importance map over the input image showing which receptive fields of the final convolutional layer contributed most to the predicted class logit. Given the high relevance attributed to the centra region displaying a female model, it seems that the model was paying attention to an discriminative feature, but none-theless misclassified the document as a scientific publication instead of an advertisement.
*Interpretation.* In correct cases, the attribution follows class-defining layout cues: dense text blocks and page frame for *scientific publication* and strong stroke/line structure for *handwritten*.
In the failure cases, both backbones confuse *advertisement* with *scientific publication*; Grad-CAM highlights high-contrast, title-like typography and centered foreground structure, suggesting a shortcut based on global layout and scan artifacts.

*Practical takeaway.* Across Grad-CAM (@fig-doc-cls-attrib-gradcam) and Integrated Gradients (@fig-doc-cls-attrib-ig), saliency concentrates on document structure rather than semantics.
Robustness could be improved by augmentations that decorrelate labels from borders/background (e.g., border randomization/cropping, stronger background perturbations, explicit margin masking).

#figure(
  caption: [Integrated Gradients (IG) attribution for a correct AlexNet test sample (target = predicted class logit).],
)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 12pt,
    row-gutter: 10pt,

    [#figure(
      image(attr_root + "/alexnet/ig_heat_best_scientific_alexnet_heatmap.png", width: 100%),
      caption: [IG heatmap — AlexNet (best) — GT+Pred: *scientific publication* (p = 0.0635).],
    ) <fig-doc-cls-attrib-ig-heatmap>],
    [#figure(
      table(
        columns: (auto, 1fr),
        align: (left, left),
        stroke: 0.5pt,
        inset: 6pt,
        [*Ground truth*], [13 = *scientific publication*],
        [*Prediction*], [13 = *scientific publication*],
        [*Confidence*], [0.0635 (≈ 1/16)],
        [*Explained target*], [predicted class logit (`target=sample.pred`)],
      ),
      caption: [Sample metadata for @fig-doc-cls-attrib-ig-heatmap.],
    ) <fig-doc-cls-attrib-ig-metadata>],

    [#figure(
      [
        IG integrates gradients along a path from a *baseline* image to the input.
        Our baseline is all zeros by default (BaselineStrategy.ZERO).
        We visualize IG with `sign="all"`: green denotes positive contributions (increasing the *scientific publication* logit) and red denotes negative contributions.
      ],
      caption: [How to read IG (red vs green).],
    ) <fig-doc-cls-attrib-ig-howto>],
    [#figure(
      [
        - *Layout-driven evidence dominates:* page frame/margins and large rectangular text blocks carry the most coherent signal.
        - *Speckled pattern ⇒ weak, unstable evidence:* the confidence is near-uniform over 16 classes, so the logit is almost flat and IG becomes noisy.
        - *Ink texture vs background:* positive contributions concentrate inside dense text regions, while negative contributions often sit in lighter margins/border regions.

        *Bottom line:* The model is correct, but not decisive; the prediction is driven primarily by page-level layout statistics rather than semantics.
      ],
      caption: [Interpretation of @fig-doc-cls-attrib-ig-heatmap.],
    ) <fig-doc-cls-attrib-ig-interpretation>],
  )
] <fig-doc-cls-attrib-ig>

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
