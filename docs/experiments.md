````markdown
# Experiments and Evaluation

## Multimodal Hate Speech Detection with Sarcasm-Aware Transformer Fusion

This document defines the experimental setup, training strategy, evaluation protocol, and ablation studies for the project.

The project methodology is the primary source of truth.

---

# 1. Experimental Objective

The experiments evaluate whether sarcasm-aware multimodal fusion improves hate-speech detection compared with simpler multimodal and unimodal approaches.

The primary research focus is the effect of:

- Sarcasm-aware gating
- Cross-modal attention
- Visual-text alignment
- OCR information
- Multimodal image-text reasoning

---

# 2. Training Strategy

Training is divided into three phases.

## Phase 1 — Modality-Specific Pretraining

The image and text encoders are initially kept frozen.

The methodology specifies:

```text
Image Encoder: ViT
Text Encoder: HateBERT
````

Only the projection layers and sarcasm module are trained during this phase.

Purpose:

* Preserve pretrained representations
* Prevent catastrophic forgetting
* Train the modality projections
* Establish the sarcasm representation

Training configuration:

```text
Epochs: 5
Learning Rate: 1e-4
```

---

## Phase 2 — Fusion Module Training

The top four transformer layers of each encoder are unfrozen.

The following components are trained:

* Image encoder upper layers
* Text encoder upper layers
* Cross-modal fusion module
* Sarcasm gate

The methodology specifies:

```text
Learning Rate: 2e-5
Warmup Scheduler
Gradient Clipping: 1.0
```

The training data consists of the combined existing and collected datasets.

---

## Phase 3 — End-to-End Fine-Tuning

The complete model is fine-tuned end-to-end.

Adversarial debiasing is introduced during this phase using a:

```text
Domain-Adversarial Neural Network (DANN)
```

The adversarial component is used to reduce dataset-specific and annotator-related bias.

Final learning rate:

```text
5e-6
```

---

# 3. Training Infrastructure

The methodology specifies the following infrastructure:

```text
Framework:
PyTorch

Model Library:
HuggingFace Transformers

Batch Size:
32

Gradient Accumulation:
×4

Precision:
fp16 mixed precision

GPU:
A100 / V100

Experiment Tracking:
Weights & Biases

Early Stopping:
Patience = 5

Cross Validation:
K = 5
```

---

# 4. Model Objective

The model jointly optimizes hate detection, sarcasm detection, target-group prediction, and visual-text alignment.

The total training objective is:

```text
L_total =
    L_hate(BCE)
    + λ1 L_sarcasm(BCE)
    + λ2 L_contrastive(NT-Xent)
    + λ3 L_target(multi-BCE)
```

The methodology specifies:

```text
λ1 = 0.3
λ2 = 0.1
λ3 = 0.2
```

The weights are tuned using the validation set.

The contrastive loss is applied to CLIP image-text embedding pairs before fusion.

---

# 5. Primary Evaluation Metric

## AUROC

Area Under the Receiver Operating Characteristic Curve is the primary evaluation metric.

It measures the ability of the classifier to distinguish hateful and non-hateful samples across classification thresholds.

---

# 6. Additional Evaluation Metrics

## Macro F1

Macro F1 is used to account for class imbalance between hate and not-hate classes.

---

## Balanced Accuracy

Balanced accuracy is calculated by averaging the per-class accuracy.

This provides a more reliable measure when the classes are imbalanced.

---

## False Positive Rate

False Positive Rate is specifically monitored because benign sarcastic content should not be incorrectly classified as hate.

---

## Cohen's Kappa

Cohen's Kappa measures agreement related to sarcasm labelling.

It is included as a measure of human-model agreement for sarcasm-related predictions.

---

## GradCAM IoU

GradCAM IoU evaluates the alignment between model attention maps and annotated hateful regions.

This provides an interpretability-oriented evaluation of whether the model focuses on relevant visual regions.

---

# 7. Evaluation Metrics Summary

| Metric              | Purpose                             |
| ------------------- | ----------------------------------- |
| AUROC               | Primary hate/not-hate evaluation    |
| Macro F1            | Handles class imbalance             |
| Balanced Accuracy   | Measures per-class performance      |
| False Positive Rate | Measures incorrect hate predictions |
| Cohen's Kappa       | Measures sarcasm-label agreement    |
| GradCAM IoU         | Measures visual-region alignment    |

---

# 8. Ablation Studies

Ablation experiments are used to determine the contribution of individual components of the proposed architecture.

---

## 8.1 Without Sarcasm Module

```text
Full Model
    ↓
Remove Sarcasm Module
    ↓
Baseline Fusion Model
```

Purpose:

Determine whether explicit sarcasm modelling improves hate-speech detection.

---

## 8.2 Without Cross-Attention

The cross-modal attention mechanism is removed.

Fusion is replaced with:

```text
Image Features
+
Text Features
      ↓
Concatenation
      ↓
Classifier
```

Purpose:

Measure the contribution of cross-modal attention.

---

## 8.3 Without Contrastive Loss

The NT-Xent contrastive alignment loss is removed.

The model performs direct classification without the contrastive alignment objective.

Purpose:

Measure the contribution of visual-text alignment learning.

---

## 8.4 Text-Only Model

Only the text modality is used.

The methodology specifies:

```text
HateBERT
```

as the text-only baseline.

Purpose:

Measure how much performance comes from textual information alone.

---

## 8.5 Image-Only Model

Only the image modality is used.

The methodology specifies:

```text
ViT + Hate Head
```

as the image-only baseline.

Purpose:

Measure the contribution of visual information independently.

---

## 8.6 Without OCR Text

OCR-derived text is removed.

The model uses:

```text
Image
+
Post Title
```

instead of the full OCR-enhanced text representation.

Purpose:

Measure the contribution of text embedded directly inside meme images.

---

# 9. Ablation Study Summary

| Experiment           | Modification                   | Purpose                        |
| -------------------- | ------------------------------ | ------------------------------ |
| Full Model           | Complete proposed architecture | Main system                    |
| w/o Sarcasm          | Remove sarcasm module          | Measure sarcasm contribution   |
| w/o Cross-Attention  | Concatenation fusion           | Measure cross-modal attention  |
| w/o Contrastive Loss | Remove NT-Xent                 | Measure alignment contribution |
| Text-Only            | HateBERT                       | Measure text-only performance  |
| Image-Only           | ViT + hate head                | Measure image-only performance |
| w/o OCR              | Image + post title             | Measure OCR contribution       |

---

# 10. Main Experiment

The main experiment compares the complete proposed model against the ablation configurations.

The primary comparison should focus on:

```text
AUROC
Macro F1
Balanced Accuracy
False Positive Rate
```

Additional analysis should include:

```text
Cohen's Kappa
GradCAM IoU
```

---

# 11. Expected Experimental Analysis

The experiments should determine:

1. Whether sarcasm-aware gating improves hate detection.
2. Whether cross-modal attention improves image-text reasoning.
3. Whether contrastive alignment improves multimodal representation quality.
4. Whether OCR provides useful information beyond the post title.
5. Whether multimodal input performs better than individual modalities.
6. Whether the proposed architecture reduces false positive hate predictions on sarcastic content.
7. Whether model attention aligns with annotated hateful regions.

These are evaluation questions, not predetermined results.

---

# 12. Result Recording

Actual experimental results should be recorded after model training.

Use a table of the following form:

| Experiment           | AUROC | Macro F1 | Balanced Accuracy | FPR | Cohen's Kappa | GradCAM IoU |
| -------------------- | ----: | -------: | ----------------: | --: | ------------: | ----------: |
| Full Model           |   TBD |      TBD |               TBD | TBD |           TBD |         TBD |
| w/o Sarcasm          |   TBD |      TBD |               TBD | TBD |           TBD |         TBD |
| w/o Cross-Attention  |   TBD |      TBD |               TBD | TBD |           TBD |         TBD |
| w/o Contrastive Loss |   TBD |      TBD |               TBD | TBD |           TBD |         TBD |
| Text-Only            |   TBD |      TBD |               TBD | TBD |           TBD |         TBD |
| Image-Only           |   TBD |      TBD |               TBD | TBD |           TBD |         TBD |
| w/o OCR              |   TBD |      TBD |               TBD | TBD |           TBD |         TBD |

Do not replace `TBD` with estimated or fabricated values.

---

# 13. Cross-Validation

The methodology specifies:

```text
K = 5
```

for cross-validation.

The same evaluation protocol should be maintained across folds.

Results should be recorded consistently for every experimental configuration.

---

# 14. Early Stopping

Early stopping is used during training.

The methodology specifies:

```text
Patience = 5
```

Training should stop when the monitored validation performance fails to improve according to the implementation's selected validation criterion.

---

# 15. Experiment Reproducibility

Each experiment should record:

```text
Experiment name
Dataset version
Configuration file
Training phase
Learning rate
Batch size
Number of epochs
Random seed
Model configuration
Loss configuration
Evaluation metrics
Checkpoint
Final results
```

Experiment tracking is performed using:

```text
Weights & Biases
```

---

# 16. Experimental Configuration

Training and evaluation settings should be stored in configuration files rather than hard-coded in experiment scripts.

Relevant configuration files include:

```text
configs/base.yaml
configs/dataset.yaml
configs/model.yaml
configs/training.yaml
configs/evaluation.yaml
```

---

# 17. Result Interpretation

The full model should be evaluated against all ablation configurations.

An improvement in AUROC or Macro F1 alone should not be treated as sufficient evidence of improvement.

The analysis should also consider:

* Balanced Accuracy
* False Positive Rate
* Sarcasm agreement
* Visual explanation quality

This is especially important because the project focuses on sarcastic hate, where visual-text incongruity can cause standard unimodal models to miss hateful meaning.

---

# 18. Experimental Limitations

The methodology does not provide final experimental results.

Therefore, this document does not claim that the proposed architecture has already achieved a specific performance level.

Numerical results must be added only after the corresponding experiments have actually been executed.

---

# 19. Source-of-Truth Rule

The project methodology is the primary source of truth for experimental design.

Implementation should not silently change:

* Training phases
* Learning rates
* Evaluation metrics
* Loss weights
* Ablation configurations
* Cross-validation strategy
* Early stopping configuration

Any intentional deviation should be documented and approved by the team.

---

# 20. Summary

The experimental framework evaluates the proposed sarcasm-aware multimodal transformer through:

```text
Training
   ↓
Validation
   ↓
5-Fold Cross-Validation
   ↓
Primary Evaluation
   ↓
Ablation Studies
   ↓
Interpretability Analysis
   ↓
Final Comparison
```

The main goal is to determine whether sarcasm-aware cross-modal reasoning provides measurable improvements for multimodal hate-speech detection.

````
