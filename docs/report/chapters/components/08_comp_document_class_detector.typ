#import "@preview/supercharged-hm:0.1.2": *
#import "@preview/wrap-it:0.1.1": wrap-content

#let ext_link_blue = rgb("#1a5fb4")
#let blink(dest, body) = link(dest)[
  #set text(fill: ext_link_blue)
  #body
]

== Document Classifier (RVL-CDIP) <comp_doc_cls>
*Jan Duchscherer*

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


==== Data, preprocessing & augmentations <doc-cls-data-prep>

For grayscale normalization (in [0, 1]), we compute dataset statistics on the RVL-CDIP train split via `DocDataModule.compute_grayscale_mean_std()`:
$mu = 0.911966$ and $sigma = 0.241507$ (single channel). These are the defaults in `TransformConfig`.

RVL-CDIP consists of grayscale document page images where *layout* is often the strongest cue.

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

Inspecting 4 sample figures in @fig-doc-cls-test-examples, we notice that the "documents" exhibit features that are atypical for modern document scans. This lets us suspect that models trained on this dataset may not generalization well to real-world documents.

*Data loading (Hugging Face).* We load RVL-CDIP from the Hugging Face Hub (`chainyo/rvl-cdip`) using `load_dataset(...)` and cache it locally (see `RVLCDIPConfig` in #blink("https://github.com/luroess/traenslenzor/blob/master/traenslenzor/doc_classifier/data_handling/huggingface_rvl_cdip_ds.py")[`traenslenzor/doc_classifier/data_handling/huggingface_rvl_cdip_ds.py`]).
Transforms are attached through `HFDataset.set_transform(...)` with a pickleable batch wrapper (`_TransformApplier`) to enable Lightning DataLoader multiprocessing.
For fast iteration and Optuna sweeps, `DocDataModule` supports deterministic subsampling via `limit_num_samples` (shuffle #sym.arrow.r select prefix) in `traenslenzor/doc_classifier/lightning/lit_datamodule.py`.

*Layout-preserving resize+pad.* All pipelines start with:
- `LongestMaxSize(max_size=img_size)` to scale the longer side to a fixed bound, preserving aspect ratio.
- `PadIfNeeded(min_height=img_size, min_width=img_size)` to obtain a square canvas without cropping.

This avoids destroying layout cues that turned out as failure modes for center-crop and rescale-only stategies that we have initially employed.

*Channel handling and normalization.* The dataset is grayscale, but torchvision backbones (ResNet/ViT) are typically ImageNet-pretrained on RGB.
Therefore, `TransformConfig` supports (a) replicating grayscale to 3 channels (`convert_to_rgb=true`) and (b) selecting the normalization statistics:
- `normalization_mode="dataset"` uses a single-channel mean/std estimated from the RVL-CDIP train split (computed by `DocDataModule.compute_grayscale_mean_std()`).
- `normalization_mode="imagenet"` uses ImageNet constants for better alignment with pretrained backbones.

#figure(
  caption: [Config-as-Factory wiring for the classifier training stack.],
)[
  #image("/graphics/doc-classifier-config.svg", width: 100%)
] <fig-doc-classifier-config-diagram>

@fig-doc-classifier-config-diagram illustrates the main components of the `doc_classifier` module's training stack and showcases the _config-as-factory_ pattern.

*Augmentation pipelines.* We provide five transform presets as Config-as-Factory targets in #blink("https://github.com/luroess/traenslenzor/blob/master/traenslenzor/doc_classifier/data_handling/transforms.py")[`traenslenzor/doc_classifier/data_handling/transforms.py`]:
- `TrainTransformConfig`: "document-friendly" heavy augmentation for scratch training (mild rotation/scale/shift, perspective, scanner-like noise/blur, brightness/contrast, gamma).
- `TrainHeavyTransformConfig`: stronger regularization for continued training (affine + distortions, compression/downscale artifacts, CLAHE, coarse/grid dropout).
- `FineTuneTransformConfig`: light augmentation for pretrained models (resize/pad + mild photometric jitter).
- `FineTunePlusTransformConfig`: moderate fine-tuning augmentation (small rotation/perspective + mild noise/blur + photometric jitter).
- `ValTransformConfig`: deterministic preprocessing for evaluation (resize/pad + channel/normalization only).

Concretely, `TrainTransformConfig` keeps geometry close to the original page ($<=$5° rotation, $<=$5% scale, $<=$2% shift) while simulating capture noise (Gaussian noise, mild blur, brightness/contrast, gamma); Starting of with stronger geometic distortions as would be typical for CV tasks on natural images caused our models to perform much worse in our initial experiments.
`TrainHeavyTransformConfig` increases geometric variability ($<=$7° rotation + affine/shear + distortion) and adds harder degradations (downscale, CLAHE, and dropout occlusions) to steer the model towards learning a variety of robust features.

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


==== Training results and analysis <doc-cls-training-results>

We report the final epoch's train/val losses and accuracies (last logged to W&B) and the best validation loss for the top runs.

#figure(
  caption: [Document Class Detector training runs on RVL-CDIP.],
)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    stroke: 0.5pt,
    inset: 6pt,
    table.header(
      [*Backbone*], [*Epoch*], [*Train acc*], [*Val acc*], [*Train loss*], [*Val loss*], [*Test acc*], [*Test loss*]
    ),
    [ALEXNET], [15], [0.954], [0.992], [0.144], [0.030], [0.992], [0.030],
    [RESNET50], [6], [0.976], [0.912], [0.078], [0.347], [0.958], [0.142],
    [VIT_B16], [9], [0.607], [0.622], [1.313], [1.276], [0.650], [1.139],
  )
] <tbl-doc-cls-training-runs>

@tbl-doc-cls-training-runs indicates rather unexpected results: the AlexNet-from-scratch model outperforms both ImageNet-pretrained backbones (ResNet-50 and ViT-B/16) by a wide margin (test accuracy 99.2% vs 95.8% / 65.0%).

Our initial hypothesis is that this gap is primarily driven by

+ AlexNet was trained end-to-end with the same learning rate for all layers, furthermore it was trained over 15 epochs, while the other two models were only trained for 10 epochs.
+ ResNet and ViT were fine-tuned, with their backbones being frozen for the first two epochs and then receiving a reduced learning rate (scale 0.01) compared to the head.
+ We also suggest that the domain shift form ImageNet to RVL-CDIP might imply that the rich set of features learned by both pretrained backbones are not well suited for document images.
+ Finally, we used the `TrainTransformConfig` augmentation pipeline for AlexNet, switching to the even heavier augmentation pipeline at the middel of the training run (epoch 8), while the pretrained backbones used the lighter `FineTunePlusTransformConfig`, which might have led to less robust features.
+ The stark underperformance of ViT-B/16 beyond ResNet-50 is most likely due to the fact that we have used optimized hyperparameters for both AlexNet and ResNet-50, while ViT was only trained in a few manual experiments without performing a full hyperparameter sweep.

To answer these questions with some more fidelity, we analyze training dynamics, per-class performance, and hyperparameter sweep results below.


==== Training dynamics and OneCycleLR schedule


#figure(
  caption: [Training curves and OneCycleLR schedule (WandB).],
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

The plots in Figure 14 summarize how the models train over `trainer/global_step`:
- *Validation accuracy / loss:* AlexNet improves steadily and practiacally reaches saturation, while the pretrained backbones plateau earlier and at a worse objective.
- *Training loss:* the step-wise loss decreases with characteristic stochastic noise from minibatch sampling and data augmentation; the gap to validation curves highlights generalization differences between runs. While the higher learning rate used to train the fine-tuned ResNet lead to an initial rapid decrease in training loss, it also seems to have contributed to the model getting stuck in an early suboptimal minimum, where it plaeaued in terms of validation loss. The ResNet's training loss exhibits a noticable drop in its training loss when the backbone is unfrozen at epoch 2, but this does not translate to an equivalent improvement in validation performance.
- *OneCycleLR schedule:* learning rates follow a warmup to `max_lr` and then decay; this schedule stabilizes early training and encourages convergence in later steps. We used n warmup of 30% for the pretrained networks and 15% for AlexNet.

==== Per-class performance (confusion matrices)

#figure(
  caption: [Row-normalized confusion matrices for selected runs.],
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



=== Optuna Hyperparameter Sweeps <doc-cls-optuna-sweeps>

#let optuna = json("/analysis/optuna_summary.json")
#let optuna_stats = json("/analysis/optuna_stats.json")
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
The objective is the validation loss (cross-entropy); all studies are minimized. All trials run for up to 6 epochs with early pruning based on intermediate validation loss on a reduced subset of the training data (20%) to speed up evaluation.

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

This sweep targets the ResNet-50 fine-tune regime and focuses on learning-rate schedule, regularization via weight decay, the type of data augmentation, the statistics used for normalization.

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
- `normalization_mode`: `dataset`
- `max_lr`: #fmt(p_resnet.at("module_config.scheduler.max_lr"))
- `pct_start`: #fmt(p_resnet.at("module_config.scheduler.pct_start"))
- `backbone_lr_scale`: #fmt(p_resnet.at("module_config.optimizer.backbone_lr_scale"))
- `weight_decay`: #fmt(p_resnet.at("module_config.optimizer.weight_decay"))
- `backbone_unfreeze_at_epoch`: #p_resnet.at("trainer_config.callbacks.backbone_unfreeze_at_epoch")

*Notes (interpretation).*
- The best configuration uses `finetune_plus` with early backbone adaptation (`backbone_unfreeze_at_epoch = 2`), consistent with the hypothesis that RVL-CDIP benefits from stronger domain-specific augmentation and feature adaptation.
- With OneCycleLR, the effective backbone peak learning rate is `max_lr * backbone_lr_scale`, here approximately #fmt(p_resnet.at("module_config.scheduler.max_lr") * p_resnet.at("module_config.optimizer.backbone_lr_scale")). This keeps backbone updates conservative while allowing the head to train at the full `max_lr`.
- The top trials are tightly clustered (best value #r3(study_resnet.best_value)), suggesting a relatively flat optimum under the fixed trial budget.

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
    The importance proxy indicates that the OneCycleLR peak learning rate (`max_lr`) and schedule shape (`pct_start`) dominate this sweep, with regularization and unfreezing acting as secondary knobs. The `transform_config` and `normalization_mode` categorical parameters unfortunately do not appear here, since they were not included in all trials and hence are not part of the importance calculation.
  ],
)

==== ResNet sweep: categorical effects

#let resnet_tests = optuna_stats.at("doc-classifier").at("categorical_tests")
#let norm_test = resnet_tests.filter(it => it.param == "normalization_mode").first()
#let transform_test = resnet_tests.filter(it => it.param == "transform_config").first()
#let unfreeze_test = resnet_tests.filter(it => it.param == "backbone_unfreeze_at_epoch").first()

#let norm_groups = norm_test.group_stats.map(it => str(it.group)).join(", ")
#let transform_groups = transform_test.group_stats.map(it => str(it.group)).join(", ")
#let unfreeze_groups = unfreeze_test.group_stats.map(it => str(it.group)).join(", ")

This sweep contains many PRUNED trials that cannot be compared directly with completed trials.
Among finite COMPLETE trials, we observe that `normalization_mode` and `transform_config` have quickly been settled on `dataset` and `finetune_plus`, respectively. The latter indicates that stronger augmentations result in a more robust model after training for only 6 epochs at 20% of the full dataset size.
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

*Notes.* The box plots and group statistics suggest that `finetune_plus` is the only transform preset that yields a sizable set of finite COMPLETE trials (n = #transform_test.group_stats.first().n) with low spread (median #r3(transform_test.group_stats.first().median), std #r3(transform_test.group_stats.first().std)). Similarly, `dataset` normalization yields the best median objective (#r3(norm_test.group_stats.first().median)) among finite COMPLETE trials (n = #norm_test.group_stats.first().n). Both findings align with the best trial's configuration.
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

*Notes.*
- The sweep is prune-heavy (trials: #study_alexnet.n_trials, complete: #study_alexnet.n_complete, pruned: #study_alexnet.n_pruned), indicating that many configurations diverge or underperform early under the fixed epoch budget.


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
    - The importance proxy ranks `max_lr` as dominant (consistent with OneCycleLR sensitivity), followed by `weight_decay` and then dropout; this aligns with the typical failure mode of too aggressive learning rates causing unstable optimization.
    - The best parameters sit near the lower end for dropout and in a narrow learning-rate regime, suggesting that capacity-limited AlexNet benefits more from stable optimization than from strong regularization.
    \
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

*Notes.* The LOWESS curves visually support a narrow "safe" region for `max_lr` (too large values correlate with higher objectives / pruning) and a milder effect for `weight_decay`. Howeever, given the limited number of finite COMPLETE trials, these trends should be interpreted with extreme caution. This is especially true for `weight_decay` where the LOWESS trend connects widely spread data points.

==== Cross-study interpretation and takeaways

- The ResNet sweep's best finite trials use dataset normalization and `finetune_plus` transforms; this supports our earlier findings where we naively normalized our data on ImageNet statistics and got extremely poor results.
- Backbone unfreezing shows only small differences and appears second-order relative to augmentation/normalization and learning-rate schedule.
- The AlexNet sweep shows the strongest sensitivity to `max_lr` and `weight_decay`, consistent with the importance plot and the LOWESS trends.
- Generalization of the sweeps' findings to longer training and full dataset size is uncertain; follow-up experiments should validate the best configurations under production conditions.


=== Attributability and Interpretability <doc-cls-attrib>

We qualitatively inspect the classifier's decision cues using Captum-based attributions @captum.
For each backbone, we scan the test split and select the most confident correct prediction ("best") and the most confident incorrect prediction ("worst").
We compute attributions for the predicted class (i.e., the explanation targets the model decision, not the ground truth).

We export multiple methods (Integrated Gradients, Input x Gradient, Noise Tunnel, Occlusion, DeepLift, LayerGradXActivation). In this report we focus on two views:
- *Grad-CAM overlays* (coarse, convolutional feature-level evidence; intuitive for layout cues).
- *Integrated Gradients (IG)* (input-level sensitivity; useful to spot border/background artifacts).

// *Grad-CAM.* Grad-CAM builds a coarse importance map on the last convolutional features by weighting each feature map by the global-average pooled gradient of the target logit and then upsampling to input resolution @gradcam:
// $ L^c = op("ReLU")(sum_k alpha_k^c A^k) $ with $ alpha_k^c = 1/(H W) sum_(i,j) partial y^c / partial A^k_(i,j) $.
// Because these feature maps are low-resolution, the overlays appear blocky; this is expected and still informative at the layout level.

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

*How to read the plots.* Grad-CAM (used for the ResNet-50 samples) builds a spatial importance map on a late convolutional layer and upsamples it to input resolution @gradcam. The overlay appears blocky because the underlying feature maps are low-resolution; this is expected and still informative at the page-layout level. IG can appear speckled when the model is uncertain (e.g., when the softmax distribution is close to uniform over 16 classes), in which case the attribution signal is weak and unstable.

*Observations (best cases).* In the best examples, the attribution concentrates on class-defining *layout* cues:
- *Scientific publication (AlexNet best, IG):* evidence spreads over the dense text block and page frame rather than local semantics.
- *Handwritten (ResNet-50 best, Layer Gradient x Activation ):* evidence follows stroke-like regions and line structure, which are discriminative for this class.

*Observations (worst cases).* In the worst examples, both backbones confuse *advertisement* with *scientific publication*. Grad-CAM highlights high-contrast, title-like typography and centralized foreground structure and scan/capture artifacts (frame, margins), rather than content-level semantics. The AlexNet attributions are non-specific, diffuse and nearly uniform, which suggests that the model failed to find discriminative features and fell back to a near-random guess. The ResNet-50 attributions focus on a distinct center-region: a depiction of a feamle model, as well as the high-contrast, non-layout-aligned title, which both appear as strong visual cues _against_ typical scientific publication layouts, hence the misclassification is rather unexpected in this case.


The qualitative evidence in @fig-doc-cls-attrib-gradcam supports the broader RVL-CDIP pattern: the classifier relies heavily on document layout. To reduce the shown failure mode, augmentations that decorrelate labels from borders/background (random crops, margin masking, stronger background/contrast perturbations) should make the model less sensitive to shortcuts and improve robustness.


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

#wrap-content(
  align: top + right,
  columns: (2fr, 3fr),
  column-gutter: 24pt,

  [

    - End-to-end Lightning training stack with modular backbones (AlexNet, ResNet-50, ViT-B/16) and model-aware preprocessing.
    - Config-as-Factory composition with TOML export/import and typed CLI overrides.
    - Experiment tooling: W&B logging, Optuna sweeps, optional Lightning tuning, and Captum interpretability hooks.
    - Generalization: models trained on RVL-CDIP exhibit nearly non-existent generalization capabilities to well-scanned modern documents converted to gray-scale. @fig-doc-classifier-chat shows an example where a nicely scanned document is m
    - Serving path as FastMCP tool integrated into the session-based File Server architecture.
  ],
  [
    #figure(
      caption: [Classfication within the trÄnslenzor system.],
    )[
      #image("/imgs/doc-scanner/chat_super-res-classify.png", width: 86%)
    ] <fig-doc-classifier-chat>
  ],
)
