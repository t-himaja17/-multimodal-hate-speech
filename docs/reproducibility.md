---

# 1. Reproducibility Objective

The objective is to ensure that experiments can be repeated using the same:

- Dataset configuration
- Model configuration
- Training configuration
- Evaluation configuration
- Random seeds
- Hardware settings
- Software environment
- Checkpoints
- Experiment tracking information

No final experimental result should be reported unless the corresponding experiment has actually been executed.

---

# 2. Project Configuration

The project uses separate configuration files for major experimental components.

```text
configs/
├── base.yaml
├── dataset.yaml
├── model.yaml
├── training.yaml
└── evaluation.yaml
````

The configuration files should contain the parameters required to reproduce the corresponding experiment.

---

# 3. Dataset Reproducibility

Every experiment should record the exact dataset version used.

The following information should be recorded:

```text
Dataset name
Dataset version
Source
Number of samples
Class distribution
Train split
Validation split
Test split
Preprocessing configuration
OCR configuration
Augmentation configuration
```

The dataset split strategy must remain consistent across experiments.

---

# 4. Data Preprocessing

The preprocessing pipeline includes:

```text
Raw Data
   ↓
Dataset Construction
   ↓
Image Filtering
   ↓
OCR + Text Processing
   ↓
Image Feature Extraction
   ↓
Text Feature Extraction
   ↓
Model Input
```

OCR processing should preserve relevant text information and positional information where available.

The methodology specifies OCR using:

```text
EasyOCR + TrOCR
```

OCR output may include:

* Extracted text
* Bounding boxes
* Font/region information
* Confidence scores

---

# 5. Model Reproducibility

The proposed architecture contains:

```text
Image Encoder
      +
Text Encoder
      +
Sarcasm Gate
      ↓
Projection & Alignment
      ↓
Cross-Modal Co-Attention
      ↓
Sarcasm Gating
      ↓
Multimodal Fusion
      ↓
Classification Heads
```

The methodology specifies:

```text
Image Encoder:
ViT

Text Encoder:
HateBERT

Shared Fusion Dimension:
512

Cross-Modal Attention:
4 layers

Attention Heads:
8

Dropout:
0.1
```

The image representation uses ViT features and CLIP visual embeddings.

The text representation uses HateBERT features and CLIP text embeddings.

---

# 6. Sarcasm Module

The sarcasm module is an important component of the proposed architecture.

The model produces a sarcasm representation that is used to modulate multimodal fusion.

The gating operation is:

```text
h_fused = h_fused ⊙ σ(W · g + b)
```

where:

```text
g = sarcasm gate representation
W = learned transformation
b = bias
σ = sigmoid
```

The sarcasm gate is used as a learned architectural control signal for multimodal reasoning.

---

# 7. Training Reproducibility

Training is divided into three phases.

## Phase 1

Modality-specific pretraining with frozen image and text encoders.

```text
Learning Rate:
1e-4
```

Only the projection layers and sarcasm module are trained.

---

## Phase 2

Fusion module training with the upper layers of the encoders unfrozen.

```text
Learning Rate:
2e-5

Warmup:
Enabled

Gradient Clipping:
1.0
```

---

## Phase 3

End-to-end fine-tuning.

```text
Learning Rate:
5e-6
```

Adversarial debiasing is introduced during this phase.

---

# 8. Training Environment

The methodology specifies:

```text
Framework:
PyTorch

Model Library:
HuggingFace Transformers

Batch Size:
32

Gradient Accumulation:
4

Precision:
fp16 mixed precision

GPU:
A100 / V100

Experiment Tracking:
Weights & Biases
```

The exact hardware used for a final experiment should be recorded.

---

# 9. Random Seeds

Every experiment should record the random seed used.

The seed should be applied consistently to:

```text
Python
NumPy
PyTorch
DataLoader
Model initialization
Data splitting
```

Recommended experiment record:

```text
Random Seed: TBD
```

The actual value must be filled in when the experiment is executed.

---

# 10. Loss Configuration

The total training objective is:

```text
L_total =
    L_hate
    + λ1 L_sarcasm
    + λ2 L_contrastive
    + λ3 L_target
```

The methodology specifies:

```text
λ1 = 0.3
λ2 = 0.1
λ3 = 0.2
```

The individual losses are:

```text
Hate Loss:
BCE

Sarcasm Loss:
BCE

Contrastive Loss:
NT-Xent

Target Group Loss:
Multi-BCE
```

---

# 11. Evaluation Reproducibility

The evaluation protocol uses the following metrics:

```text
AUROC
Macro F1
Balanced Accuracy
False Positive Rate
Cohen's Kappa
GradCAM IoU
```

AUROC is the primary evaluation metric.

Macro F1 and Balanced Accuracy are used to account for class imbalance.

False Positive Rate is particularly important for evaluating false hate predictions on benign sarcastic content.

GradCAM IoU is used for visual-region alignment analysis.

---

# 12. Cross-Validation

The methodology specifies:

```text
5-Fold Cross-Validation
K = 5
```

The same evaluation procedure should be applied across all folds.

Dataset splitting must not introduce data leakage between training, validation, and test sets.

---

# 13. Early Stopping

Early stopping is used during training.

```text
Patience = 5
```

The validation criterion used for early stopping should be recorded with each experiment.

---

# 14. Experiment Tracking

Experiments should be tracked using:

```text
Weights & Biases
```

Each experiment should record:

```text
Experiment name
Dataset version
Configuration files
Git commit
Random seed
Training phase
Learning rate
Batch size
Number of epochs
Loss configuration
Validation metric
Best checkpoint
Final checkpoint
Evaluation results
```

---

# 15. Checkpoint Management

Each completed experiment should have an identifiable checkpoint.

Recommended structure:

```text
checkpoints/
├── phase1/
├── phase2/
└── phase3/
```

Checkpoint metadata should include:

```text
Model configuration
Training phase
Dataset version
Random seed
Epoch
Validation score
Git commit
```

---

# 16. Git Version Tracking

The Git commit associated with an experiment should be recorded.

This allows the exact implementation used for an experiment to be identified.

Recommended experiment record:

```text
Git Commit:
TBD
```

The actual commit hash should be recorded after the implementation is committed.

---

# 17. Configuration Tracking

Every experiment must record the configuration files used.

```text
configs/base.yaml
configs/dataset.yaml
configs/model.yaml
configs/training.yaml
configs/evaluation.yaml
```

Configuration changes should be committed to Git.

Avoid changing configuration parameters without recording the change.

---

# 18. Ablation Reproducibility

The following ablation experiments should use the same evaluation protocol as the full model:

```text
Full Model

Without Sarcasm Module

Without Cross-Attention

Without Contrastive Loss

Text-Only

Image-Only

Without OCR
```

Only the intended component should be changed for each ablation whenever possible.

This ensures that performance differences can be attributed to the component being evaluated.

---

# 19. Reproducibility Checklist

Before running a final experiment, verify:

```text
[ ] Dataset version recorded
[ ] Dataset split recorded
[ ] Configuration files committed
[ ] Random seed recorded
[ ] Git commit recorded
[ ] Model configuration recorded
[ ] Training configuration recorded
[ ] Evaluation configuration recorded
[ ] Hardware recorded
[ ] Software environment recorded
[ ] Experiment tracked
[ ] Checkpoint saved
[ ] Evaluation results saved
```

---

# 20. Result Integrity

Experimental results must be based on actual model execution.

Do not:

* Invent numerical results
* Estimate missing metrics
* Replace failed experiments with assumed values
* Change evaluation settings after seeing results without documenting the change
* Report an experiment that was not executed

Any change to the methodology or experimental configuration must be documented.

---

# 21. Source-of-Truth Rule

The project methodology is the primary source of truth.

The reproducibility documentation exists to record how the methodology is implemented and evaluated.

The relationship is:

```text
Project Methodology
        ↓
Module Interface Contract
        ↓
Configuration
        ↓
Implementation
        ↓
Experiment
        ↓
Evaluation
        ↓
Recorded Results
```

If an implementation intentionally deviates from the methodology, the deviation must be documented and approved by the team.

---

# 22. Final Reproducibility Record

For every final experiment, maintain the following record:

```text
Experiment:
TBD

Dataset:
TBD

Dataset Version:
TBD

Git Commit:
TBD

Random Seed:
TBD

Configuration:
TBD

Training Phase:
TBD

Hardware:
TBD

Best Checkpoint:
TBD

AUROC:
TBD

Macro F1:
TBD

Balanced Accuracy:
TBD

False Positive Rate:
TBD

Cohen's Kappa:
TBD

GradCAM IoU:
TBD
```

`TBD` values must only be replaced after the corresponding experiment has been executed.

---

# 23. Summary

The project is reproducible when the following chain is preserved:

```text
Dataset
   ↓
Preprocessing
   ↓
Configuration
   ↓
Model
   ↓
Training
   ↓
Checkpoint
   ↓
Evaluation
   ↓
Results
```

All important configuration, implementation, and experiment changes should be version controlled using Git.

````
