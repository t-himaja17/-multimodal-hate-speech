````markdown
# Module Interface Contract

This document defines the interfaces between the major modules of the Multimodal Hate Speech Detection — Sarcasm-Aware Transformer Fusion system.

The project methodology is the primary source of truth. Any implementation detail not explicitly specified by the methodology is marked as `TBD` and must be agreed upon by the team before implementation.

---

# 1. End-to-End Pipeline

```
Dataset Construction
        ↓
OCR + Text Processing
        ↓
Image Feature Extraction
        ↓
Text Feature Extraction
        ↓
Sarcasm-Aware Detection
        ↓
Projection & Alignment
        ↓
Cross-Modal Co-Attention
        ↓
Sarcasm Gating
        ↓
Classification Heads
        ↓
Predictions + Explainability
````

The final system produces:

* Hate / Not-Hate prediction
* Target-group prediction
* Severity score
* Explainability map

---

# 2. Dataset Interface

## Dataset Sources

The methodology uses existing benchmarks together with newly scraped data.

Existing datasets include:

* Hateful Memes Challenge
* MultiOFF
* iSarcasm
* MUStARD
* HateSPAN
* MemeCap

Additional data sources include:

* Reddit
* Twitter/X
* Know Your Meme metadata

## Processing

The dataset pipeline includes:

1. Dataset collection
2. Image filtering
3. Perceptual-hash deduplication
4. Meme-template filtering
5. OCR extraction
6. Dual annotation
7. Class balancing
8. Data augmentation

## Annotation

Each annotated sample contains:

```text
hate / not-hate
target group
sarcasm / irony flag
incongruity type
```

Incongruity types include:

```text
visual-text conflict
hyperbole
exaggeration
```

The methodology specifies:

```text
3 annotators
majority voting
MTurk / Label Studio
Cohen's kappa >= 0.70
```

## Output

The data module must provide a standardized sample containing:

```text
sample ID
image
OCR text
OCR regions
post title / caption context
labels
metadata
```

Exact serialization format: TBD.

---

# 3. OCR Interface

## Input

```text
Meme image
```

## Models

```text
EasyOCR
+
TrOCR
```

## OCR Information

The system stores:

```text
OCR text
bounding boxes
font / region information
confidence scores
```

## Positional Tags

OCR text is represented using:

```text
[TOP]
[BOTTOM]
[CAPTION]
```

## Output

```text
OCR text
+
OCR bounding boxes
+
region labels
+
confidence scores
```

## Additional Processing

The text pipeline handles:

* code-switching
* internet slang
* emoji → text conversion
* abbreviations
* post title
* KYM template description

Location:

```text
src/multimodal_hate/data/
```

Owner:

```text
Member 1
```

---

# 4. Data Augmentation Interface

## Input

```text
Annotated training samples
```

## Image Augmentation

```text
horizontal flip
colour jitter
text-region masking
```

## Text Augmentation

```text
English → German → English
```

using MarianMT.

## Class Balancing

```text
SMOTE on feature embeddings
```

## Output

```text
Balanced and augmented training samples
```

Location:

```text
src/multimodal_hate/data/
```

Owner:

```text
Member 1
```

---

# 5. Image Feature Extraction Interface

## 5.1 ViT-L/14

### Input

```text
Preprocessed meme image
```

### Model

```text
ViT-L/14
```

### Output

```text
[CLS] token + patch embeddings
1024-dimensional representation
```

Logical interface:

```text
[B, N_img, 1024]
```

---

# 6. CLIP Visual Encoder Interface

## Input

```text
Preprocessed meme image
```

## Model

```text
CLIP Visual Encoder
ViT-B/32
```

## Output

```text
512-dimensional image embedding
```

Logical interface:

```text
[B, 512]
```

---

# 7. Supplementary Visual Features

The methodology specifies:

```text
YOLOv8
```

for object bounding boxes such as:

```text
weapons
symbols
```

It also specifies:

```text
ArcFace / DeepFace
```

for facial emotion recognition.

The exact combined supplementary feature dimension is:

```text
TBD
```

---

# 8. Text Feature Extraction Interface

## 8.1 HateBERT

### Model

```text
HateBERT
BERT fine-tuned on RAL-E
```

### Input

```text
Structured OCR text
+
template caption / metadata
```

### Output

```text
[CLS] representation
768 dimensions
```

Logical interface:

```text
[B, N_txt, 768]
```

---

# 9. Secondary Text Encoders

The methodology specifies:

```text
RoBERTa-large
+
CLIP Text Encoder
```

RoBERTa provides robust linguistic representations.

CLIP Text Encoder produces text embeddings in the shared image-text space.

CLIP text output:

```text
512-dimensional embedding
```

Logical interface:

```text
[B, 512]
```

---

# 10. Handcrafted Linguistic Features

The methodology specifies:

```text
Slur presence
Negation detection
Intensifiers
Punctuation density
LIWC sentiment categories
```

The slur lexicon is based on HateXplain.

## Output

```text
64-dimensional feature appendage
```

Logical interface:

```text
[B, 64]
```

Location:

```text
src/multimodal_hate/models/features/
```

Owner:

```text
Member 2
```

---

# 11. Sarcasm Module

The sarcasm module combines:

```text
Textual sarcasm
Visual-text incongruity
Sentiment reversal
```

and produces a learned sarcasm gate.

Location:

```text
src/multimodal_hate/models/sarcasm/
```

Owner:

```text
Member 3
```

---

# 12. Sarcasm Encoder

## Model

```text
SarcasmBERT
```

Fine-tuned on:

```text
iSarcasm
SARC 2.0
```

## Input

```text
OCR text
+
template caption
```

## Output

```text
sarcasm probability
+
span markers
```

---

# 13. Visual-Text Incongruity Interface

## Input

```text
CLIP image embedding
+
CLIP text embedding
```

## Method

```text
CLIP cosine similarity
+
learned incongruity head
```

## Output

```text
Incongruity score ∈ [0, 1]
```

---

# 14. Sentiment Reversal Interface

## Method

```text
VADER
+
BERT sentiment
```

## Input

Text regions:

```text
TOP
BOTTOM
```

## Output

```text
Polarity flip indicator
```

---

# 15. Sarcasm Gate Interface

## Input

```text
SarcasmBERT output
+
Visual-text incongruity
+
Sentiment reversal information
```

## Processing

```text
Concatenation
      ↓
MLP
      ↓
Sigmoid gating
```

## Output

```text
Sarcasm gate vector g ∈ R^d
```

The exact dimension `d` is:

```text
TBD
```

---

# 16. Projection and Alignment Interface

The shared fusion dimension is:

```text
d_fuse = 512
```

## Image Projection

```text
Input:
[B, N_img, 1024]

Output:
[B, N_img, 512]
```

## Text Projection

```text
Input:
[B, N_txt, 768]

Output:
[B, N_txt, 512]
```

Processing:

```text
Linear projection
+
LayerNorm
+
learnable modality type embeddings
```

Modality embeddings:

```text
IMG_TOKEN
TXT_TOKEN
```

Location:

```text
src/multimodal_hate/models/fusion/
```

Owner:

```text
Member 4
```

---

# 17. Sarcasm-Conditioned Attention Bias

The methodology specifies adding the sarcasm gate as an additive bias to cross-attention logits.

Conceptually:

```text
attention_logits'
=
attention_logits
+
sarcasm_bias(g)
```

The exact broadcasting/projection implementation is:

```text
TBD
```

---

# 18. Cross-Modal Co-Attention

## Number of Layers

```text
4
```

## Attention Heads

```text
8
```

## Dropout

```text
0.1
```

Components:

```text
Image → Text Cross-Attention
Text → Image Cross-Attention
Self-Attention within Fused Stream
```

## Image → Text

Image tokens attend to text tokens.

## Text → Image

Text tokens attend to image patches.

## Fused Self-Attention

The concatenated sequence:

```text
[IMG ; TXT]
```

attends to itself.

## Output

```text
Fused multimodal representation
```

Location:

```text
src/multimodal_hate/models/fusion/
```

Owner:

```text
Member 4
```

---

# 19. Sarcasm Gating of Fused Representation

The methodology specifies:

```text
h_fused' =
h_fused ⊙ σ(W · g + b)
```

## Processing

```text
Linear layer
      ↓
Sigmoid
      ↓
Element-wise multiplication
```

## Output

```text
Sarcasm-conditioned fused representation
```

---

# 20. Contrastive Alignment Interface

## Method

```text
NT-Xent loss
```

## Input

```text
CLIP image embedding
+
CLIP text embedding
```

The positive pair consists of the image and text belonging to the same meme.

## Output

```text
contrastive loss
```

---

# 21. Classification Heads

## 21.1 Hate / Not-Hate

Architecture:

```text
2-layer MLP
↓
Softmax
```

Output:

```text
Hate
Not-Hate
```

Loss:

```text
BCE
```

---

# 22. Target Group Head

Separate sigmoid heads are used for:

```text
Race
Religion
Gender
Disability
Sexuality
```

Activation:

```text
Sigmoid
```

Loss:

```text
Multi-label BCE
```

---

# 23. Sarcasm Auxiliary Head

## Input

```text
Sarcasm gate intermediate features
```

## Output

```text
Binary sarcasm prediction
```

## Loss

```text
BCE
```

## Loss Weight

```text
λ₁ = 0.3
```

---

# 24. Severity Score

The methodology lists severity score as a final system output.

However, it does not specify the exact:

```text
severity labels
severity head architecture
severity output dimension
severity loss
```

Therefore:

```text
Severity implementation = TBD
```

---

# 25. Total Loss

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

Weights:

```text
λ₁ = 0.3
λ₂ = 0.1
λ₃ = 0.2
```

Losses:

```text
L_hate        = BCE
L_sarcasm     = BCE
L_contrastive = NT-Xent
L_target      = Multi-label BCE
```

---

# 26. Training Interface

## Phase 1 — Modality-Specific Pretraining

Freeze:

```text
ViT backbone
HateBERT backbone
```

Train:

```text
Projection layers
Sarcasm module
```

Settings:

```text
Epochs = 5
Learning rate = 1e-4
```

## Phase 2 — Fusion Module Training

Unfreeze:

```text
Top 4 transformer layers of each encoder
```

Train:

```text
Cross-attention fusion
Sarcasm gate
```

Settings:

```text
Learning rate = 2e-5
Warmup scheduler
Gradient clipping = 1.0
```

## Phase 3 — End-to-End Fine-Tuning + Adversarial Debiasing

Fine-tune:

```text
Full model
```

Add:

```text
Adversarial classifier
```

Method:

```text
DANN-style adversarial loss
```

Final learning rate:

```text
5e-6
```

---

# 27. Training Infrastructure

```text
PyTorch
HuggingFace Transformers
Batch size = 32
Gradient accumulation = 4
Mixed precision = fp16
A100 / V100 GPU
Weights & Biases logging
Early stopping patience = 5
K-fold cross-validation K = 5
```

---

# 28. Evaluation Interface

Metrics:

```text
AUROC
Macro F1
Balanced Accuracy
False Positive Rate
Cohen's Kappa
GradCAM IoU
```

Location:

```text
src/multimodal_hate/evaluation/
```

Owner:

```text
Member 4
```

---

# 29. Explainability Interface

Methods:

```text
GradCAM
Spatial attention
Cross-modal attention
```

Output:

```text
Explainability map
```

The exact GradCAM implementation layer is:

```text
TBD
```

---

# 30. Ablation Interface

The methodology specifies:

### A. Without sarcasm module

```text
Baseline fusion only
```

### B. Without cross-attention

```text
Concatenation fusion
```

### C. Without contrastive loss

```text
Direct classification
```

### D. Text-only

```text
HateBERT
```

### E. Image-only

```text
ViT + hate head
```

### F. Without OCR text

```text
Image + post-title only
```

---

# 31. Inference Interface

## Input

```text
Meme image
+
optional post-title/context
```

## Processing

```text
Image
  ↓
OCR
  ↓
Structured OCR Text
  ↓
Image + Text Encoders
  ↓
Sarcasm Module
  ↓
Projection & Alignment
  ↓
Cross-Modal Co-Attention
  ↓
Sarcasm Gating
  ↓
Classification Heads
  ↓
Prediction + Explainability
```

## Output

```text
Hate / Not-Hate
Target Group
Severity Score
Sarcasm Score
Explainability Map
```

Exact serialized inference output:

```text
TBD
```

Location:

```text
src/multimodal_hate/inference/
```

---

# 32. Public Module Interfaces

## Member 1 — Data

```python
load_dataset(...)
prepare_sample(...)
run_ocr(...)
build_dataset_record(...)
create_data_splits(...)
```

## Member 2 — Encoders

```python
encode_image_vit(...)
encode_image_clip(...)
encode_text_hatebert(...)
encode_text_clip(...)
encode_text_roberta(...)
extract_linguistic_features(...)
```

## Member 3 — Sarcasm

```python
predict_sarcasm(...)
compute_incongruity(...)
detect_sentiment_reversal(...)
compute_sarcasm_gate(...)
```

## Member 4 — Fusion / Training / Evaluation

```python
project_modalities(...)
cross_modal_fusion(...)
apply_sarcasm_gate(...)
compute_total_loss(...)
train_one_step(...)
evaluate_model(...)
run_inference(...)
```

---

# 33. Tensor Interface Summary

## ViT

```text
[B, N_img, 1024]
```

## HateBERT

```text
[B, N_txt, 768]
```

## CLIP Image

```text
[B, 512]
```

## CLIP Text

```text
[B, 512]
```

## Handcrafted Features

```text
[B, 64]
```

## Projected Image

```text
[B, N_img, 512]
```

## Projected Text

```text
[B, N_txt, 512]
```

## Fusion Dimension

```text
512
```

## Sarcasm Gate

```text
[B, d]
```

where:

```text
d = TBD
```

---

# 34. Team Dependency Contract

## Member 1 → Member 2

Member 1 provides:

```text
Image
OCR text
OCR regions
Post title / caption
Dataset labels
Metadata
```

## Member 2 → Member 3

Member 2 provides:

```text
CLIP image embedding
CLIP text embedding
Structured text representation
```

## Member 2 + Member 3 → Member 4

Member 2 provides:

```text
Image features
Text features
CLIP image embedding
CLIP text embedding
```

Member 3 provides:

```text
Sarcasm probability
Incongruity score
Sentiment reversal
Sarcasm gate
```

---

# 35. Interface Rules

1. Do not silently change a tensor dimension.
2. Do not silently change a public function signature.
3. Document every public input and output.
4. Use configuration files for dataset paths.
5. Use configuration files for model hyperparameters.
6. Do not hard-code API keys or secrets.
7. Do not commit datasets directly to Git.
8. Do not commit large model checkpoints directly to Git.
9. Add tests for major modules.
10. Test module interfaces using dummy inputs before full integration.
11. Changes to shared interfaces require team agreement.
12. Methodology parameters must not be changed silently.
13. Every experiment must record its configuration.
14. Every experiment must record its results.
15. `main` represents the stable integrated implementation.

---

# 36. TBD Items

```text
Exact tokenizer maximum length
Exact image preprocessing resolution
Exact N_img
Exact N_txt
Exact sarcasm gate dimension d
Exact RoBERTa output used downstream
Exact supplementary visual-feature dimension
Exact final fusion pooling operation
Exact severity labels
Exact severity head
Exact severity loss
Exact DANN loss weighting
Exact protected-attribute proxy representation
Exact heterogeneous-label mapping
Exact missing-label handling
Exact train/validation/test split strategy
```

These values must be agreed upon by the team before they become required interfaces.

---

# 37. Source-of-Truth Rule

The project methodology is the primary source of truth.

```text
Project Methodology
        ↓
Module Interface Contract
        ↓
Implementation
```
If an implementation intentionally deviates from the methodology, the deviation must be documented and approved by the team.

