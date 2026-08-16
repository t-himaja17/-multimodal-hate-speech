````markdown
# Project Methodology

## Multimodal Hate Speech Detection in Memes with Sarcasm-Aware Transformer Fusion

This document records the project's methodology and acts as the primary technical source of truth for implementation.

The methodology covers dataset construction, multimodal feature extraction, sarcasm-aware detection, transformer-based multimodal fusion, classification, training, evaluation, ablation studies, and the research contributions of the proposed system.

---

# 1. End-to-End Methodology Pipeline

```
Data Collection
      ↓
Data Processing
      ↓
Image / Text Feature Extraction
      ↓
Sarcasm Module
      ↓
Sarcasm-Aware Transformer Fusion
      ↓
Classification
      ↓
Predictions + Explainability
````

The six major stages are:

1. Data Collection
2. Data Processing
3. Feature Extraction
4. Sarcasm Module
5. Fusion
6. Output

The final system produces:

* Hate / Not-Hate prediction
* Target-group prediction
* Severity score
* Explainability map

---

# 2. Dataset Construction

## 2.1 Existing Datasets

### Hateful Memes Challenge

The Hateful Memes Challenge dataset is used as a multimodal hate-detection benchmark.

It contains approximately:

* 10,000 memes
* Binary hate labels
* Confounder sets

The dataset is used as a gold-standard multimodal benchmark.

---

### MultiOFF

MultiOFF is a multimodal offensive-content dataset.

It contains:

* Approximately 750 memes
* Offensive / not-offensive labels
* Image and overlaid-text information

---

### iSarcasm + MUStARD

These datasets provide sarcasm-related supervision.

iSarcasm provides sarcasm labels.

MUStARD provides multimodal sarcasm information.

For this project, the methodology specifies adapting the text + image portions where applicable.

---

### HateSPAN + MemeCap

HateSPAN provides span-level hateful-token annotations.

MemeCap provides rich captions for meme images and is used for semantic grounding of visual content.

---

# 3. Additional Data Collection

## 3.1 Reddit

Reddit data is collected using PRAW.

Target subreddits include:

* r/dankmemes
* r/PoliticalHumor
* r/ComedyCemetery
* r/ControversialHumor

The collection process:

```text
Reddit posts
    ↓
Select image posts
    ↓
Extract image
    ↓
Extract post title
    ↓
Use title as meme-caption context
```

Estimated collection:

```text
15K–25K samples
```

---

## 3.2 Twitter / X

Twitter/X data is collected using the Academic API methodology specified in the project.

Collection uses:

* Hate-associated hashtags
* Target-group keywords
* Relevant keyword lists

Estimated collection:

```text
~10K tweets
```

---

## 3.3 Know Your Meme

Know Your Meme metadata is used to obtain:

* Meme template descriptions
* Template context
* Visual grounding information

This information is combined with OCR text and post context.

---

# 4. Scraping and Annotation Pipeline

The complete data preparation pipeline is:

```text
Keyword / Hashtag Seeding
        ↓
Image Filtering
        ↓
Perceptual Hash Deduplication
        ↓
Meme Template Filtering
        ↓
OCR Extraction
        ↓
Dual Annotation
        ↓
Class Balancing
        ↓
Data Augmentation
```

---

# 5. Keyword and Hashtag Seeding

Seed terms are defined for target groups such as:

* Race
* Religion
* Gender
* Sexuality

The methodology specifies using:

* HateXplain slur lexicons
* Custom expansion
* WordNet
* Embedding nearest neighbors

---

# 6. Image Filtering

Only image posts are retained.

Perceptual-hash deduplication is applied.

The methodology specifies:

```text
pHash similarity threshold ≥ 95%
```

Non-meme images are filtered using a meme-template classifier.

The specified classifier is:

```text
ResNet-50
fine-tuned on Know Your Meme data
```

---

# 7. OCR Processing

OCR is performed using an ensemble of:

* EasyOCR
* TrOCR

The system stores:

* OCR text
* Bounding boxes
* Font / region information
* Confidence scores

Text regions are positionally tagged:

```text
[TOP]
[BOTTOM]
[CAPTION]
```

OCR text is merged with post-title information as structured context.

---

# 8. Dual Annotation

The project uses simultaneous hate and sarcasm annotation.

Three annotators are specified.

Annotation platforms:

* MTurk
* Label Studio

Each sample is annotated for:

### Hate

```text
hate
not-hate
```

### Target Group

```text
race
religion
gender
sexuality
```

### Sarcasm

```text
sarcasm / irony flag
```

### Incongruity Type

```text
visual-text conflict
hyperbole
exaggeration
```

Majority voting is used.

The target inter-annotator agreement is:

```text
Cohen's κ ≥ 0.70
```

---

# 9. Class Balancing and Augmentation

## 9.1 Class Balancing

SMOTE is applied to feature embeddings for minority classes.

---

## 9.2 Image Augmentation

The methodology specifies:

* Horizontal flipping
* Colour jitter
* Text-region masking

Text-region masking follows a SimCLR-style augmentation approach.

---

## 9.3 Text Augmentation

Back-translation is used:

```text
English
  ↓
German
  ↓
English
```

The specified translation system is:

```text
MarianMT
```

---

# 10. Image Feature Extraction

The image branch contains multiple visual representations.

## 10.1 ViT-L/14

ViT-L/14 is used as the primary vision backbone.

It uses patch-based image encoding.

The model captures:

* Global semantic features
* Meme-template structure
* Image layout
* Visual relationships

The methodology specifies:

```text
14 × 14 patches
```

Output:

```text
[CLS] token + patch embeddings
```

with:

```text
1024-dimensional representation
```

---

## 10.2 CLIP Visual Encoder

The visual CLIP encoder is:

```text
ViT-B/32
```

CLIP provides image embeddings aligned with language representations.

Output:

```text
512-dimensional shared image embedding
```

CLIP is also used for:

* Cross-modal similarity
* Visual-text incongruity
* Zero-shot template recognition

---

## 10.3 Supplementary Visual Features

The methodology additionally specifies:

### YOLOv8

Used for object bounding boxes such as:

* Weapons
* Symbols

### ArcFace / DeepFace

Used for facial emotion recognition.

These outputs are concatenated into an auxiliary feature vector.

---

## 10.4 Spatial Attention

Learnable spatial attention weights identify important visual regions.

Examples include:

* Symbols
* Gestures
* Other hateful visual regions

Cross-modal attention from text tokens guides the spatial attention.

The attention maps are also used for GradCAM explainability.

---

# 11. Text Feature Extraction

## 11.1 HateBERT

HateBERT is the primary text encoder.

The methodology describes it as:

```text
BERT fine-tuned on RAL-E
```

It is specialized for:

* Slurs
* Coded language
* Dog-whistles
* Implicit hate

Output:

```text
[CLS] representation
768 dimensions
```

---

## 11.2 RoBERTa-large

RoBERTa-large provides additional robust linguistic representations.

---

## 11.3 CLIP Text Encoder

The CLIP text encoder generates text embeddings in the shared image-text embedding space.

These embeddings enable direct comparison with CLIP image embeddings.

---

# 12. Structured OCR Text

OCR text is processed using positional region tags:

```text
[TOP]
[BOTTOM]
[CAPTION]
```

The text processing handles:

* Code-switching
* Internet slang
* Emoji conversion
* Abbreviations

Emoji conversion uses:

```text
emot
```

OCR text is combined with the Know Your Meme template description.

---

# 13. Handcrafted Linguistic Features

The methodology specifies the following auxiliary features:

* Slur presence
* Negation detection
* Intensifiers
* Punctuation density
* LIWC sentiment categories

The slur lexicon is based on HateXplain.

These features form a:

```text
64-dimensional feature appendage
```

which is concatenated with the transformer representation.

---

# 14. Sarcasm-Aware Detection Module

Sarcasm is treated as a core component of the architecture rather than only as an auxiliary prediction task.

The purpose is to detect cases where:

```text
positive / neutral text
        +
negative / hateful image
        ↓
visual-text incongruity
        ↓
possible sarcastic hate
```

The module contains five components.

---

# 15. Sarcasm Encoder

The specified model is:

```text
SarcasmBERT
```

fine-tuned using:

* iSarcasm
* SARC 2.0

Input:

```text
OCR text
+
template caption
```

Output:

```text
sarcasm probability
+
span markers
```

The encoder identifies cues such as:

* Hyperbole
* Irony markers
* Negated sentiment

---

# 16. Visual-Text Incongruity

The visual-text incongruity component uses:

```text
CLIP cosine similarity
+
learned incongruity head
```

Input:

```text
CLIP image embedding
+
CLIP text embedding
```

Output:

```text
Incongruity score ∈ [0, 1]
```

The score represents semantic mismatch between the image and text.

This is a core sarcasm signal.

---

# 17. Sentiment Reversal Detector

The methodology specifies:

```text
VADER
+
BERT sentiment
```

The analysis is performed separately for text regions such as:

```text
TOP
BOTTOM
```

Output:

```text
Polarity flip indicator
```

This is intended to capture patterns such as positive captions paired with negative visual content.

---

# 18. Sarcasm Gate Vector

The outputs of the sarcasm components are combined.

The methodology specifies:

```text
Sarcasm outputs
      ↓
Concatenation
      ↓
MLP
      ↓
Sigmoid
      ↓
Sarcasm gate vector g
```

The gate is:

```text
g ∈ R^d
```

where the exact dimension is determined by the fusion implementation.

The gate is used to modulate the fusion transformer.

---

# 19. Sarcasm Auxiliary Loss

The sarcasm head uses:

```text
Binary Cross Entropy (BCE)
```

Input:

```text
Gate intermediate features
```

Output:

```text
Sarcasm binary prediction
```

The sarcasm loss weight is:

```text
λ₁ = 0.3
```

---

# 20. Sarcasm-Aware Transformer Fusion

The fusion transformer is the core architectural contribution.

The input consists of:

```text
Image embeddings
+
Text embeddings
+
Sarcasm gate
```

---

# 21. Projection and Alignment

The shared fusion dimension is:

```text
d_fuse = 512
```

Image and text representations are projected into the same dimension.

Processing:

```text
Image features
      ↓
Linear projection
      ↓
LayerNorm
      ↓
512-dimensional representation
```

and:

```text
Text features
      ↓
Linear projection
      ↓
LayerNorm
      ↓
512-dimensional representation
```

Learnable modality-type embeddings are added:

```text
IMG_TOKEN
TXT_TOKEN
```

The design is inspired by ViLBERT.

---

# 22. Sarcasm-Conditioned Attention Bias

The sarcasm gate is added as a bias to the cross-attention logits.

Conceptually:

```text
Cross-attention logits
        +
Sarcasm gate bias
        ↓
Sarcasm-conditioned attention
```

A high sarcasm score increases the influence of image-text conflict during fusion.

---

# 23. Cross-Modal Co-Attention

The fusion module contains:

```text
4 layers
8 attention heads
dropout = 0.1
```

Three attention operations are specified.

## Image → Text

Image tokens attend to text tokens.

This learns which image patches are contextualized by which words.

---

## Text → Image

Text tokens attend to image patches.

This produces visually grounded text representations.

---

## Fused Self-Attention

The image and text tokens are concatenated:

```text
[IMG ; TXT]
```

The combined sequence attends to itself using standard multi-head self-attention.

This enables holistic multimodal reasoning.

---

# 24. Sarcasm Gating of Fused Representation

After fusion, the sarcasm gate modulates the fused representation.

The specified operation is:

```text
h_fused'
=
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

This produces a sarcasm-conditioned fused representation.

---

# 25. Contrastive Alignment

The methodology uses:

```text
NT-Xent loss
```

between:

```text
CLIP image embedding
+
CLIP text embedding
```

for corresponding image-text pairs.

The purpose is to preserve image-text alignment structure while allowing semantic mismatch to remain informative.

Contrastive loss weight:

```text
λ₂ = 0.1
```

---

# 26. Classification Heads

## 26.1 Hate / Not-Hate

Primary classification head:

```text
2-layer MLP
      ↓
Softmax
```

Classes:

```text
Hate
Not-Hate
```

Loss:

```text
Binary Cross Entropy
```

Optional class weighting may be used for imbalance.

---

## 26.2 Target Group

The target-group prediction is multi-label.

Separate sigmoid heads are used for:

```text
Race
Religion
Gender
Disability
Sexuality
```

---

## 26.3 Sarcasm Auxiliary Head

A separate sarcasm head predicts the sarcasm label.

Loss:

```text
BCE
```

Weight:

```text
λ₁ = 0.3
```

---

# 27. Total Training Objective

The methodology defines:

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

with:

```text
λ₁ = 0.3
λ₂ = 0.1
λ₃ = 0.2
```

Loss functions:

```text
L_hate        = BCE
L_sarcasm     = BCE
L_contrastive = NT-Xent
L_target      = Multi-label BCE
```

The methodology states that the loss weights are tuned through grid search on the validation set.

Contrastive loss is applied to CLIP embedding pairs before fusion.

---

# 28. Training Strategy

Training occurs in three phases.

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

Purpose:

```text
Prevent catastrophic forgetting
```

Settings:

```text
Epochs = 5
Learning rate = 1e-4
```

---

## Phase 2 — Fusion Module Training

Unfreeze:

```text
Top 4 transformer layers
```

of each encoder.

Train:

```text
Cross-attention fusion
+
Sarcasm gate
```

Training data:

```text
Existing datasets
+
Scraped dataset
```

Settings:

```text
Learning rate = 2e-5
Warmup scheduler
Gradient clipping = 1.0
```

---

## Phase 3 — End-to-End Fine-Tuning + Adversarial Debiasing

Fine-tune the full model.

Add an adversarial classifier operating on protected-attribute proxies related to annotator identity.

Use:

```text
Domain-Adversarial Neural Network (DANN)
```

to reduce demographic / dataset-specific bias.

Final learning rate:

```text
5e-6
```

---

# 29. Training Infrastructure

The specified infrastructure is:

```text
PyTorch
HuggingFace Transformers
```

Training configuration:

```text
Batch size = 32
Gradient accumulation = 4
Mixed precision = fp16
```

Hardware:

```text
A100 / V100 GPU
```

Experiment tracking:

```text
Weights & Biases
```

Training controls:

```text
Early stopping
Patience = 5
```

Validation:

```text
5-fold cross-validation
```

---

# 30. Evaluation Protocol

The methodology specifies the following evaluation metrics.

## AUROC

Primary metric for the Hateful Memes benchmark.

---

## Macro F1

Used to handle class imbalance between hate and not-hate.

---

## Balanced Accuracy

Average of per-class accuracy.

---

## False Positive Rate

Important for avoiding benign sarcastic content being incorrectly classified as hate.

---

## Cohen's Kappa

Used to evaluate agreement related to sarcasm labeling.

---

## GradCAM IoU

Measures alignment between attention / explanation maps and annotated hateful regions.

---

# 31. Ablation Studies

The methodology specifies six ablation experiments.

## A. Without Sarcasm Module

```text
Baseline fusion only
```

---

## B. Without Cross-Attention

Replace cross-attention with:

```text
Concatenation fusion
```

---

## C. Without Contrastive Loss

Use:

```text
Direct classification
```

without the contrastive objective.

---

## D. Text-Only

Use:

```text
HateBERT
```

as the primary modality.

---

## E. Image-Only

Use:

```text
ViT + hate head
```

---

## F. Without OCR Text

Use:

```text
Image + post-title
```

without OCR text.

---

# 32. Research Contributions

The methodology identifies five main contributions.

## 1. Sarcasm as a Dynamic Attention Modulator

Sarcasm is not treated only as a side task.

The proposed sarcasm gate directly modulates cross-modal attention weights inside the fusion transformer.

This allows the model to dynamically re-weight modalities based on detected incongruity.

---

## 2. Visual-Text Incongruity via CLIP

CLIP's shared image-text embedding space is used to compute visual-text semantic dissimilarity as a sarcasm signal.

This provides a visual sarcasm proxy without requiring sarcasm-specific visual labels.

---

## 3. Dual-Annotated Dataset

The combined dataset introduces simultaneous:

```text
Hate labels
+
Sarcasm labels
```

for the same meme.

This enables joint learning of:

```text
P(hate, sarcasm | meme)
```

---

## 4. Structured OCR with Positional Region Tagging

OCR text is tagged using:

```text
[TOP]
[BOTTOM]
[CAPTION]
```

and combined with Know Your Meme template metadata.

This gives the model structural awareness of meme format.

---

## 5. Adversarial Debiasing

DANN-style adversarial training is used against proxy annotator attributes.

The goal is to reduce dataset-specific and annotator-driven bias and improve generalization across platforms and communities.

---

# 33. Final Expected Processing Flow

For one meme, the system operates as follows:

```text
Meme Image
     +
Post Title / Context
     ↓
Data Processing
     ↓
OCR
     ↓
Structured OCR Text
[TOP] [BOTTOM] [CAPTION]
     ↓
┌───────────────────────┐
│                       │
│ Image Feature Branch  │
│                       │
│ ViT-L/14 + CLIP       │
│                       │
└───────────┬───────────┘
            │
            │
┌───────────▼───────────┐
│                       │
│ Text Feature Branch   │
│                       │
│ HateBERT + RoBERTa    │
│ + CLIP Text           │
│                       │
└───────────┬───────────┘
            │
            ▼
     Sarcasm Module
            │
     ┌──────┼─────────┐
     │      │         │
 Sarcasm  Visual   Sentiment
 BERT    Incongruity Reversal
     │      │         │
     └──────┼─────────┘
            ↓
      Sarcasm Gate g
            ↓
 Projection + Alignment
            ↓
 Cross-Modal Co-Attention
            ↓
    Sarcasm Gating
            ↓
   Fused Representation
            ↓
      Classification
      ┌─────┼─────┐
      ↓     ↓     ↓
    Hate  Target Sarcasm
   Head   Group   Head
      │     │     │
      └─────┼─────┘
            ↓
 Predictions + Explainability
```

---

# 34. Final System Outputs

For each input meme, the system is designed to produce:

```text
1. Hate / Not-Hate prediction

2. Target group prediction

3. Severity score

4. Sarcasm prediction

5. Explainability map
```

The severity-score implementation details are not further specified in this methodology and therefore must not be invented during implementation.

---

# 35. Source-of-Truth Rule

This methodology is the primary source of truth for the implementation.

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

# 36. Important Implementation Principle

Do not silently change:

* Model architecture
* Dataset methodology
* Loss functions
* Loss weights
* Training phases
* Evaluation metrics
* Fusion mechanism
* Sarcasm-gating mechanism

If an implementation detail is not explicitly specified by the methodology, mark it as:

```text
TBD
```

and agree on it as a team before making it a fixed interface.

````

### 5. Click **Preview**

Make sure the headings, lists, and diagrams look normal.

### 6. Commit

Use:

```text
Document project methodology
````
