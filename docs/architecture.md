````markdown
# System Architecture

## Multimodal Hate Speech Detection with Sarcasm-Aware Transformer Fusion

This document describes the technical architecture of the proposed multimodal hate speech detection system.

The project methodology is the primary source of truth. This architecture document describes how the methodology is organized into implementation modules.

---

# 1. High-Level Architecture

The complete system follows this pipeline:

```text
Raw Meme Dataset
       ↓
Dataset Loading
       ↓
OCR + Text Processing
       ↓
Image Feature Extraction
       +
Text Feature Extraction
       ↓
Sarcasm Module
       ↓
Projection & Modality Alignment
       ↓
Sarcasm-Aware Transformer Fusion
       ↓
Classification Heads
       ↓
Predictions
       +
Explainability
````

---

# 2. Major System Components

The system is divided into the following major components:

1. Dataset Loading
2. Data Preprocessing
3. OCR and Text Processing
4. Image Feature Extraction
5. Text Feature Extraction
6. Sarcasm Detection
7. Projection and Alignment
8. Multimodal Fusion
9. Classification
10. Training
11. Evaluation
12. Explainability
13. Inference

---

# 3. Dataset Layer

## Responsibility

The dataset layer is responsible for:

* Loading raw datasets
* Reading meme images
* Reading associated text
* Reading labels
* Creating train/validation/test data
* Providing samples to downstream modules

Expected raw sample information may include:

```text
image
text / caption
OCR text
hate label
target-group label
sarcasm label
metadata
```

The exact dataset schema must follow the dataset methodology and configuration.

---

# 4. Data Preprocessing Layer

The preprocessing layer prepares raw multimodal data for model consumption.

The main operations are:

```text
Raw Image
    ↓
Image preprocessing
    ↓
Image tensor
```

and:

```text
Raw Text
    ↓
Text cleaning
    ↓
OCR processing
    ↓
Structured text
```

OCR text may contain positional region information such as:

```text
[TOP]
[BOTTOM]
[CAPTION]
```

The preprocessing stage must preserve the relationship between the image and its associated text.

---

# 5. OCR and Text Processing

OCR is used to extract text embedded inside memes.

The methodology specifies an OCR ensemble consisting of:

* EasyOCR
* TrOCR

OCR processing produces:

```text
OCR text
+
bounding boxes
+
confidence information
+
region information
```

The extracted text is combined with available post-title or caption context.

The resulting structured text is passed to the text encoders.

---

# 6. Image Feature Extraction

The image branch extracts semantic and visual representations from meme images.

The methodology specifies:

## ViT-L/14

ViT-L/14 provides the primary visual representation.

Its output contains:

```text
[CLS] representation
+
patch embeddings
```

The methodology specifies a 1024-dimensional representation.

---

## CLIP Visual Encoder

The CLIP visual encoder provides an image representation in the shared image-text embedding space.

The methodology specifies:

```text
CLIP ViT-B/32
```

with a:

```text
512-dimensional
```

shared representation.

---

## Additional Visual Features

The methodology also specifies supplementary visual information from:

```text
YOLOv8
ArcFace / DeepFace
```

These may provide:

* Object information
* Bounding boxes
* Facial emotion information

These supplementary features are treated as auxiliary visual information.

---

# 7. Text Feature Extraction

The text branch produces semantic representations from OCR text and contextual text.

## HateBERT

HateBERT is the primary hate-speech-aware text encoder.

The methodology specifies a:

```text
768-dimensional
```

representation.

---

## RoBERTa-large

RoBERTa-large provides an additional language representation.

---

## CLIP Text Encoder

The CLIP text encoder produces a representation in the same shared embedding space as the CLIP image encoder.

This enables image-text similarity and incongruity calculations.

---

# 8. Linguistic Feature Layer

The methodology specifies auxiliary linguistic features including:

* Slur presence
* Negation
* Intensifiers
* Punctuation density
* LIWC sentiment categories

These form a:

```text
64-dimensional
```

feature appendage.

The exact concatenation location must follow the module interface contract.

---

# 9. Sarcasm Module

The sarcasm module is a core part of the proposed architecture.

It receives information from the text and visual branches.

The module contains:

```text
Sarcasm Encoder
       +
Visual-Text Incongruity
       +
Sentiment Reversal
       ↓
Sarcasm Gate
```

---

# 10. Sarcasm Encoder

The methodology specifies:

```text
SarcasmBERT
```

fine-tuned using sarcasm-related datasets.

Inputs include:

```text
OCR text
+
template caption
```

Outputs include:

```text
sarcasm probability
+
sarcasm-related span information
```

---

# 11. Visual-Text Incongruity

The system calculates visual-text incongruity using:

```text
CLIP image embedding
+
CLIP text embedding
```

The methodology specifies:

```text
CLIP cosine similarity
+
learned incongruity head
```

The resulting incongruity score is used as a sarcasm signal.

---

# 12. Sentiment Reversal

Sentiment reversal is detected using:

```text
VADER
+
BERT sentiment
```

The analysis is performed on relevant text regions.

The output is a polarity-flip indicator.

---

# 13. Sarcasm Gate

The outputs of the sarcasm components are combined.

The methodology specifies:

```text
Sarcasm features
      ↓
Concatenation
      ↓
MLP
      ↓
Sigmoid
      ↓
Sarcasm gate
```

The gate is represented as:

```text
g ∈ R^d
```

The exact implementation dimension must follow the agreed module interface.

---

# 14. Projection and Modality Alignment

Before multimodal fusion, image and text representations are projected into a shared fusion space.

The methodology specifies:

```text
Fusion dimension = 512
```

The general process is:

```text
Image representation
       ↓
Linear projection
       ↓
LayerNorm
       ↓
512-dimensional representation
```

and:

```text
Text representation
       ↓
Linear projection
       ↓
LayerNorm
       ↓
512-dimensional representation
```

Modality-type information is added to distinguish image and text representations.

---

# 15. Sarcasm-Aware Transformer Fusion

The fusion transformer is the core multimodal reasoning component.

It receives aligned:

```text
Image features
+
Text features
+
Sarcasm gate information
```

The methodology specifies:

```text
4 transformer layers
8 attention heads
dropout = 0.1
```

---

# 16. Cross-Modal Attention

The fusion module performs multiple forms of attention.

## Image → Text

Image tokens attend to text tokens.

This allows visual information to use linguistic context.

---

## Text → Image

Text tokens attend to image tokens.

This allows textual information to become visually grounded.

---

## Fused Self-Attention

Image and text tokens are combined:

```text
[IMAGE TOKENS ; TEXT TOKENS]
```

The combined sequence is processed through self-attention.

This enables joint multimodal reasoning.

---

# 17. Sarcasm-Conditioned Attention

The sarcasm gate is incorporated into the attention mechanism.

Conceptually:

```text
Normal attention
        +
Sarcasm gate information
        ↓
Sarcasm-aware attention
```

This allows the fusion system to give greater importance to image-text relationships when sarcasm or incongruity is detected.

---

# 18. Sarcasm Gating of Fused Representation

After multimodal fusion, the fused representation is modulated using the sarcasm gate.

The methodology specifies the operation:

```text
h_fused' =
h_fused ⊙ σ(W · g + b)
```

where:

```text
g = sarcasm gate
W = learned projection
b = bias
σ = sigmoid
⊙ = element-wise multiplication
```

The resulting representation is passed to the classification heads.

---

# 19. Classification Layer

The fused representation is passed to multiple prediction heads.

The primary outputs include:

```text
Hate / Not-Hate
Target Group
Sarcasm
```

---

# 20. Hate Classification Head

The primary hate classification head is:

```text
Fused representation
       ↓
2-layer MLP
       ↓
Hate / Not-Hate
```

The methodology specifies binary classification.

---

# 21. Target Group Head

The target-group component predicts relevant target categories.

The methodology specifies multi-label prediction.

The target groups include:

```text
Race
Religion
Gender
Disability
Sexuality
```

The exact label mapping must follow the dataset and interface configuration.

---

# 22. Sarcasm Auxiliary Head

A separate sarcasm prediction head is trained using the sarcasm representation.

The sarcasm objective uses:

```text
Binary Cross Entropy
```

The methodology specifies:

```text
λ₁ = 0.3
```

for the sarcasm auxiliary loss.

---

# 23. Training Architecture

Training occurs in three major phases.

## Phase 1

Modality-specific pretraining.

The methodology specifies freezing:

```text
ViT backbone
HateBERT backbone
```

while training:

```text
Projection layers
Sarcasm module
```

---

## Phase 2

Fusion-module training.

The methodology specifies unfreezing the top four transformer layers of each encoder and training:

```text
Cross-attention fusion
+
Sarcasm gate
```

---

## Phase 3

End-to-end fine-tuning and adversarial debiasing.

The methodology specifies:

```text
Full-model fine-tuning
+
DANN-based adversarial debiasing
```

---

# 24. Loss Architecture

The total training objective is:

```text
L_total =
L_hate
+
λ₁ L_sarcasm
+
λ₂ L_contrastive
+
λ₃ L_target
```

The specified weights are:

```text
λ₁ = 0.3
λ₂ = 0.1
λ₃ = 0.2
```

The loss components are:

```text
Hate loss
    ↓
BCE

Sarcasm loss
    ↓
BCE

Contrastive loss
    ↓
NT-Xent

Target-group loss
    ↓
Multi-label BCE
```

---

# 25. Contrastive Alignment

The contrastive objective operates on corresponding:

```text
CLIP image embedding
+
CLIP text embedding
```

The methodology specifies:

```text
NT-Xent loss
```

with:

```text
λ₂ = 0.1
```

---

# 26. Adversarial Debiasing

The final training phase incorporates:

```text
Domain-Adversarial Neural Network
```

The adversarial component is intended to reduce unwanted demographic or dataset-specific bias.

The exact protected-attribute proxy representation must follow the methodology and agreed interface.

---

# 27. Explainability Layer

The system provides explainability alongside predictions.

The methodology specifies visual explanation through:

```text
GradCAM
```

The explainability component can highlight important visual regions contributing to the model's decision.

The architecture therefore produces:

```text
Prediction
+
Visual explanation
```

rather than only a classification label.

---

# 28. Evaluation Layer

The evaluation module measures:

```text
AUROC
Macro F1
Balanced Accuracy
False Positive Rate
Cohen's Kappa
GradCAM IoU
```

The evaluation module consumes model predictions and ground-truth labels and produces evaluation metrics and reports.

---

# 29. Ablation Architecture

The project includes the following ablation configurations:

```text
A. Without Sarcasm Module

B. Without Cross-Attention

C. Without Contrastive Loss

D. Text-Only

E. Image-Only

F. Without OCR Text
```

These experiments are used to measure the contribution of individual architectural components.

---

# 30. Inference Flow

During inference, a new meme follows:

```text
Input Meme
    ↓
Image + Text Processing
    ↓
OCR
    ↓
Image Encoder
    +
Text Encoder
    ↓
Sarcasm Module
    ↓
Projection
    ↓
Sarcasm-Aware Fusion
    ↓
Classification Heads
    ↓
Predictions
    +
Explainability
```

---

# 31. Module Ownership

The implementation is divided among four team members.

| Module                                     | Owner    | Main Location                                                    |
| ------------------------------------------ | -------- | ---------------------------------------------------------------- |
| Dataset loading and preprocessing          | Member 1 | `src/multimodal_hate/data/`                                      |
| Image/Text encoders and feature extraction | Member 2 | `src/multimodal_hate/models/encoders/` and `features/`           |
| Sarcasm module                             | Member 3 | `src/multimodal_hate/models/sarcasm/`                            |
| Fusion, training and evaluation            | Member 4 | `src/multimodal_hate/models/fusion/`, `training/`, `evaluation/` |

All modules must communicate through the interfaces documented in:

```text
docs/interfaces.md
```

---

# 32. Repository-Level Architecture

The corresponding implementation structure is:

```text
src/
└── multimodal_hate/
    ├── data/
    │   ├── loaders/
    │   ├── processing/
    │   └── scraping/
    │
    ├── models/
    │   ├── encoders/
    │   ├── features/
    │   ├── sarcasm/
    │   └── fusion/
    │
    ├── training/
    ├── evaluation/
    └── inference/
```

Supporting project components are:

```text
configs/
scripts/
tests/
artifacts/
experiments/
docs/
```

---

# 33. Data Flow Between Modules

The primary data flow is:

```text
Data Module
    ↓
Processed Sample
    ↓
Encoder Module
    ↓
Image + Text Features
    ↓
Sarcasm Module
    ↓
Sarcasm Gate
    ↓
Fusion Module
    ↓
Fused Representation
    ↓
Classification Heads
    ↓
Predictions
```

Training and evaluation consume the outputs of the fusion and classification components.

---

# 34. Core vs Auxiliary Components

## Core Components

The following are central to the proposed architecture:

* Multimodal dataset
* OCR/text processing
* Image encoder
* Text encoder
* Sarcasm module
* Sarcasm gate
* Projection/alignment
* Cross-modal transformer fusion
* Hate classification
* Target-group classification
* Sarcasm auxiliary objective
* Contrastive alignment
* Evaluation

## Auxiliary / Additional Components

The methodology also specifies:

* YOLOv8 features
* ArcFace / DeepFace features
* Linguistic feature appendage
* GradCAM explainability
* DANN adversarial debiasing
* Ablation experiments

Their exact implementation must follow the methodology and interface contract.

---

# 35. Source-of-Truth Rule

The project methodology is the primary source of truth.

The implementation hierarchy is:

```text
Project Methodology
        ↓
Module Interface Contract
        ↓
Configuration
        ↓
Implementation
        ↓
Experiments
        ↓
Evaluation
```

If an implementation intentionally deviates from the methodology, the deviation must be documented and approved by the team.

---

# 36. Important Architecture Principle

Modules must remain independently testable.

A module should communicate with another module through the documented interface rather than directly depending on another member's internal implementation.

This reduces code conflicts and allows the four team members to work independently.

Any change to a shared interface must be discussed and documented before implementation.

````
