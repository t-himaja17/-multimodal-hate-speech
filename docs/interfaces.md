Yes. I re-checked the **actual methodology file** before rewriting this. The previous version I gave you was too generic in a few places. This version keeps the methodology's specific details—ViT-L/14, CLIP ViT-B/32, HateBERT, RoBERTa, 64-d handcrafted features, 512-d fusion, 4 co-attention layers, 8 heads, dropout 0.1, sarcasm components, loss weights, 3 training phases, evaluation metrics, and ablations.   

**Important:** Copy everything inside the code block and replace the current contents of `docs/interfaces.md` with it.

````markdown
# Module Interface Contract

This document defines the technical interfaces between the major modules of
the Multimodal Hate Speech Detection system with Sarcasm-Aware Transformer
Fusion.

The methodology document is the primary source for this contract.

Where the methodology specifies an exact model, dimension, loss, training
setting, or architecture parameter, that specification is preserved here.

Where the methodology does not specify an exact low-level implementation
detail, it is marked as `TBD` and must be agreed upon by the team before
implementation.

---

# 1. End-to-End Pipeline

```text
Existing Benchmarks + Scraped Data
                |
                v
        Dataset Construction
                |
                v
     Deduplication + Annotation
                |
                v
          OCR Processing
                |
                v
       Data Augmentation
       + Class Balancing
                |
                v
      Multimodal Features
        /             \
       v               v
 Image Encoders    Text Encoders
 ViT-L/14          HateBERT
 CLIP              RoBERTa
                   CLIP Text
        \             /
         \           /
          v         v
       Sarcasm Module
             |
             v
    Sarcasm Gate Vector
             |
             v
    Projection + Alignment
             |
             v
   Cross-Modal Co-Attention
             |
             v
      Fused Representation
             |
             v
     Classification Heads
       /       |        \
      v        v         v
    Hate     Target    Sarcasm
             Group     Auxiliary
             |
             v
      Predictions +
      Explainability
````

---

# 2. Module Ownership

| Module                                                   | Owner    | Main Location                                                    |
| -------------------------------------------------------- | -------- | ---------------------------------------------------------------- |
| Dataset loading, scraping support, OCR and preprocessing | Member 1 | `src/multimodal_hate/data/`                                      |
| Image/text encoders and feature extraction               | Member 2 | `src/multimodal_hate/models/encoders/` and `features/`           |
| Sarcasm-aware detection module                           | Member 3 | `src/multimodal_hate/models/sarcasm/`                            |
| Fusion, training and evaluation                          | Member 4 | `src/multimodal_hate/models/fusion/`, `training/`, `evaluation/` |

---

# 3. Dataset Construction Interface

## Input

Dataset sources may include:

* Hateful Memes Challenge
* MultiOFF
* iSarcasm
* MUStARD
* HateSPAN
* MemeCap
* Reddit
* Twitter/X
* Know Your Meme metadata

The methodology combines existing benchmarks with scraped data.

## Output

Every dataset source must eventually be converted into a common logical
sample representation.

Recommended standardized record:

```python
{
    "sample_id": str,
    "image_path": str,
    "post_title": str | None,
    "template_caption": str | None,
    "ocr_text": str,
    "ocr_regions": list,
    "hate_label": int | None,
    "target_group": dict | None,
    "sarcasm_label": int | None,
    "incongruity_type": str | None,
    "metadata": dict
}
```

The exact fields available depend on the source dataset.

## Dataset-specific information

### Hateful Memes Challenge

* Approximately 10,000 memes
* Binary hate labels
* Multimodal benchmark
* Confounder sets

### MultiOFF

* Approximately 750 memes
* Offensive / not-offensive labels
* Image and overlaid-text information

### iSarcasm + MUStARD

* Sarcasm-labelled data
* MUStARD provides multimodal sarcasm information
* Only compatible text/image portions should be adapted for this project

### HateSPAN + MemeCap

* HateSPAN provides hateful span annotations
* MemeCap provides rich captions for semantic grounding

### Reddit

The methodology specifies PRAW scraping from:

* `r/dankmemes`
* `r/PoliticalHumor`
* `r/ComedyCemetery`
* `r/ControversialHumor`

Image posts are retained and the post title is used as meme caption context.

### Twitter/X + Know Your Meme

Twitter/X is used for collecting meme-related posts and Know Your Meme
metadata is used for meme-template visual grounding.

---

# 4. Scraping Interface

## Input

```text
Platform/API credentials
Target communities
Keyword/hashtag seeds
Target-group terminology
```

## Processing

The methodology specifies:

1. Keyword and hashtag seeding
2. Image filtering
3. Perceptual-hash deduplication
4. Meme-template filtering
5. OCR extraction
6. Annotation
7. Class balancing and augmentation

## Output

```python
{
    "sample_id": str,
    "image_path": str,
    "post_title": str | None,
    "source": str,
    "metadata": dict
}
```

## Important processing details

The methodology specifies:

* pHash deduplication with a 95% similarity threshold
* Meme filtering using a ResNet-50 classifier fine-tuned on KYM
* EasyOCR + TrOCR ensemble
* OCR bounding boxes
* OCR region information
* OCR confidence scores

---

# 5. Annotation Interface

## Input

```text
Processed meme image
OCR text
Post/title context
```

## Annotation outputs

The methodology specifies annotation for:

```text
1. Hate / Not-Hate
2. Target group
3. Sarcasm / irony flag
4. Incongruity type
```

Possible incongruity categories include:

```text
- Visual-text conflict
- Hyperbole
- Exaggeration
```

## Annotation protocol

The methodology specifies:

* 3 annotators
* Majority voting
* MTurk / Label Studio
* Cohen's kappa >= 0.70 threshold

## Output

```python
{
    "hate_label": int,
    "target_group": dict,
    "sarcasm_label": int,
    "incongruity_type": str
}
```

---

# 6. Data Augmentation and Class Balancing Interface

## Input

```text
Annotated training samples
```

## Class balancing

The methodology specifies:

```text
SMOTE on feature embeddings
```

for minority-class balancing.

## Image augmentation

The methodology specifies:

```text
Horizontal flip
Colour jitter
Text-region masking
```

## Text augmentation

The methodology specifies:

```text
Back-translation:
English -> German -> English
using MarianMT
```

## Output

```text
Balanced and augmented training dataset
```

---

# 7. OCR Interface

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

The methodology specifies an EasyOCR + TrOCR ensemble.

## Output

```python
{
    "text": str,
    "regions": [
        {
            "text": str,
            "bbox": [x1, y1, x2, y2],
            "region": str,
            "confidence": float
        }
    ]
}
```

## Region tags

The methodology introduces:

```text
[TOP]
[BOTTOM]
[CAPTION]
```

These positional tags are preserved as part of structured OCR processing.

## Location

```text
src/multimodal_hate/data/processing/
```

## Owner

Member 1

---

# 8. Image Encoder Interface

## 8.1 ViT-L/14

### Input

```text
Preprocessed meme image
```

### Processing

Vision Transformer with 14 x 14 patches.

### Output

The methodology specifies:

```text
[CLS] token + patch embeddings
1024-dimensional representation
```

Logical representation:

```text
[B, N_img, 1024]
```

where:

```text
B      = batch size
N_img  = number of image tokens
1024   = feature dimension
```

The exact value of `N_img` depends on image preprocessing and is therefore:

```text
N_img = TBD
```

### Location

```text
src/multimodal_hate/models/encoders/
```

### Owner

Member 2

---

# 9. CLIP Visual Encoder Interface

## Model

```text
CLIP Visual Encoder
ViT-B/32
```

## Input

```text
Preprocessed meme image
```

## Output

```text
512-dimensional image embedding
```

Logical batch representation:

```text
[B, 512]
```

The embedding is used for:

* Cross-modal similarity
* Visual-text incongruity
* Contrastive alignment

### Location

```text
src/multimodal_hate/models/encoders/
```

### Owner

Member 2

---

# 10. Supplementary Visual Feature Interface

The methodology includes supplementary visual features.

## Object detection

```text
YOLOv8
```

Used for object bounding boxes such as:

```text
Weapons
Symbols
Other relevant objects
```

## Face / emotion information

The methodology specifies:

```text
ArcFace / DeepFace
```

for facial/emotion-related information.

## Output

```text
Auxiliary visual feature vector
```

The exact dimensionality is:

```text
TBD
```

because the methodology does not specify a fixed final dimension.

### Location

```text
src/multimodal_hate/models/features/
```

### Owner

Member 2

---

# 11. Spatial Attention / Visual Explainability Interface

The methodology specifies learnable spatial attention over image patch tokens.

## Input

```text
ViT patch embeddings
Cross-modal attention information
```

## Output

```text
Spatial attention weights
```

These weights highlight visually relevant regions and support:

```text
GradCAM explainability
```

### Location

```text
src/multimodal_hate/models/features/
```

### Owner

Member 2

---

# 12. Text Encoder Interface

## 12.1 HateBERT

### Model

```text
HateBERT
BERT fine-tuned on RAL-E
```

### Input

Structured OCR text and contextual text.

The input may include:

```text
[TOP] text
[BOTTOM] text
[CAPTION] text
KYM template description
Post/title context
```

### Output

The methodology specifies:

```text
[CLS] = 768 dimensions
```

Logical representation:

```text
[B, N_txt, 768]
```

where:

```text
B      = batch size
N_txt  = number of text tokens
768    = hidden dimension
```

### Location

```text
src/multimodal_hate/models/encoders/
```

### Owner

Member 2

---

# 13. RoBERTa + CLIP Text Interface

The methodology also specifies:

```text
RoBERTa-large
+
CLIP Text Encoder
```

## RoBERTa

Used for robust linguistic representations.

The exact final representation used in fusion is:

```text
TBD
```

unless explicitly selected during implementation.

## CLIP Text Encoder

Used to produce language embeddings in the shared image-text space.

### Output

```text
CLIP text embedding
512 dimensions
```

Logical representation:

```text
[B, 512]
```

This is used for:

* Visual-text similarity
* Incongruity detection
* Contrastive alignment

### Location

```text
src/multimodal_hate/models/encoders/
```

### Owner

Member 2

---

# 14. Structured OCR Text Processing Interface

## Input

```text
Raw OCR regions
Post/title context
KYM template metadata
```

## Processing

The methodology specifies:

```text
[TOP]
[BOTTOM]
[CAPTION]
```

region tags.

Additional processing includes:

* Code-switching handling
* Internet slang handling
* Emoji-to-text conversion
* Abbreviation handling
* KYM template-description concatenation

## Output

```text
Structured text sequence
```

### Location

```text
src/multimodal_hate/data/processing/
```

### Owner

Member 1

---

# 15. Handcrafted Linguistic Feature Interface

The methodology specifies the following auxiliary features:

```text
Slur presence
Negation detection
Intensifiers
Punctuation density
LIWC sentiment categories
```

The slur lexicon includes HateXplain-derived terminology.

## Output

```text
64-dimensional feature appendage
```

This is concatenated with the transformer `[CLS]` representation.

Therefore:

```text
[B, 64]
```

for the handcrafted feature vector.

### Location

```text
src/multimodal_hate/models/features/
```

### Owner

Member 2

---

# 16. Sarcasm Encoder Interface

## Model

```text
SarcasmBERT
BERT fine-tuned on:
- iSarcasm
- SARC 2.0
```

## Input

```text
OCR text
+
template caption
```

## Output

```python
{
    "sarcasm_probability": float,
    "span_markers": list
}
```

The methodology specifies that the module identifies textual sarcasm cues
such as:

* Hyperbole
* Irony markers
* Negated sentiment

### Location

```text
src/multimodal_hate/models/sarcasm/
```

### Owner

Member 3

---

# 17. Visual-Text Incongruity Interface

## Method

```text
CLIP cosine similarity
+
learned incongruity head
```

## Input

```text
CLIP image embedding
[B, 512]

CLIP text embedding
[B, 512]
```

## Output

```text
Incongruity score
```

with:

```text
0 <= score <= 1
```

Logical representation:

```text
[B, 1]
```

The incongruity score represents semantic mismatch between visual content
and text.

### Location

```text
src/multimodal_hate/models/sarcasm/
```

### Owner

Member 3

---

# 18. Sentiment Reversal Interface

## Method

```text
VADER
+
BERT sentiment
```

## Input

Text regions are evaluated separately:

```text
TOP region
BOTTOM region
```

## Output

```text
Polarity flip indicator
```

Logical representation:

```text
[B, 1]
```

The purpose is to detect patterns such as:

```text
positive textual sentiment
+
negative visual/contextual meaning
```

### Location

```text
src/multimodal_hate/models/sarcasm/
```

### Owner

Member 3

---

# 19. Sarcasm Gate Interface

The sarcasm gate combines:

```text
SarcasmBERT output
+
Visual-text incongruity
+
Sentiment reversal
```

## Input

Logical combined sarcasm feature representation:

```text
[B, D_sarc]
```

where:

```text
D_sarc = TBD
```

because the methodology does not specify the exact concatenated dimension
before the MLP.

## Processing

```text
Concatenation
      |
      v
MLP
      |
      v
Sigmoid gating
```

## Output

The methodology specifies:

```text
Gate vector g ∈ R^d
```

The exact `d` is not explicitly specified.

Therefore:

```text
g = [B, d]
d = TBD
```

### Location

```text
src/multimodal_hate/models/sarcasm/
```

### Owner

Member 3

---

# 20. Projection and Alignment Interface

The fusion system projects image and text representations into:

```text
d_fuse = 512
```

## Image projection

### Input

```text
[B, N_img, 1024]
```

### Output

```text
[B, N_img, 512]
```

## Text projection

### Input

```text
[B, N_txt, 768]
```

### Output

```text
[B, N_txt, 512]
```

## Processing

The methodology specifies:

```text
Linear projection
+
LayerNorm
+
learnable modality type embeddings
```

Modality tokens:

```text
IMG_TOKEN
TXT_TOKEN
```

### Location

```text
src/multimodal_hate/models/fusion/
```

### Owner

Member 4

---

# 21. Sarcasm-Conditioned Positional Bias Interface

The sarcasm gate is used to influence cross-modal attention.

## Input

```text
Sarcasm gate vector g
```

## Processing

The methodology specifies:

```text
Add g as an additive bias to cross-attention logits
```

High sarcasm/incongruity should increase the importance of relevant
image-text interaction.

## Output

```text
Sarcasm-conditioned attention logits
```

Exact implementation shape:

```text
TBD
```

because the methodology describes the operation but does not specify every
low-level broadcasting detail.

### Location

```text
src/multimodal_hate/models/fusion/
```

### Owner

Member 4

---

# 22. Cross-Modal Co-Attention Interface

## Architecture

The methodology specifies:

```text
4 cross-modal co-attention layers
```

Each layer contains:

```text
Image -> Text Cross-Attention
Text -> Image Cross-Attention
Self-Attention within fused stream
```

## Attention settings

```text
Number of layers = 4
Attention heads = 8
Dropout = 0.1
```

## Input

```text
Image tokens:
[B, N_img, 512]

Text tokens:
[B, N_txt, 512]

Sarcasm-conditioned attention bias
```

## Processing

### Image -> Text

Image tokens attend to text tokens.

### Text -> Image

Text tokens attend to image patches.

### Self-Attention

The concatenated:

```text
[IMG ; TXT]
```

sequence attends to itself.

## Output

```text
Fused token sequence
[B, N_fused, 512]
```

where:

```text
N_fused = N_img + N_txt
```

subject to the exact implementation.

### Location

```text
src/multimodal_hate/models/fusion/
```

### Owner

Member 4

---

# 23. Sarcasm Gating of Fused Representation

The methodology specifies element-wise gating:

```text
h_fused' = h_fused ⊙ σ(W · g + b)
```

## Input

```text
Fused representation
+
Sarcasm gate vector
```

## Processing

```text
Linear layer
      |
      v
Sigmoid
      |
      v
Element-wise multiplication
```

## Output

```text
Sarcasm-conditioned fused representation
```

The feature dimension remains:

```text
512
```

for the shared fusion dimension.

The exact pooling operation from the fused token sequence to the final
classification representation is:

```text
TBD
```

### Location

```text
src/multimodal_hate/models/fusion/
```

### Owner

Member 4

---

# 24. Contrastive Alignment Interface

## Method

```text
NT-Xent loss
```

## Input

Matched image-text pairs:

```text
CLIP image embedding
[B, 512]

CLIP text embedding
[B, 512]
```

## Processing

Contrastive learning encourages matching image-text pairs to be aligned
while separating mismatched pairs.

## Output

```text
contrastive_loss: scalar
```

## Loss weight

```text
lambda_2 = 0.1
```

The methodology specifies that the contrastive loss is applied to CLIP
embedding pairs before fusion.

### Location

```text
src/multimodal_hate/training/
```

### Owner

Member 4

---

# 25. Hate Classification Head

## Input

```text
Final fused representation
```

Shared fusion dimension:

```text
512
```

## Architecture

```text
2-layer MLP
    |
    v
Softmax
```

## Output

```text
Hate / Not-Hate
```

Logical class probabilities:

```text
[B, 2]
```

Classes:

```text
0 = Not-Hate
1 = Hate
```

## Loss

```text
BCE
```

Optional class weights may be used for imbalance.

### Location

```text
src/multimodal_hate/models/fusion/
```

### Owner

Member 4

---

# 26. Target Group Classification Head

The methodology specifies separate sigmoid heads for:

```text
Race
Religion
Gender
Disability
Sexuality
```

## Input

```text
Final fused representation
```

## Output

Logical multi-label representation:

```text
[B, 5]
```

Each output represents the probability of the corresponding target group.

## Activation

```text
Sigmoid
```

## Loss

```text
Multi-label BCE
```

### Location

```text
src/multimodal_hate/models/fusion/
```

### Owner

Member 4

---

# 27. Sarcasm Auxiliary Head

## Input

```text
Sarcasm gate intermediate features
```

## Output

```text
Sarcasm probability
```

Logical representation:

```text
[B, 1]
```

## Loss

```text
BCE
```

## Loss weight

```text
lambda_1 = 0.3
```

The auxiliary head is jointly trained with the main hate classification task.

### Location

```text
src/multimodal_hate/models/fusion/
```

### Owner

Member 4

---

# 28. Severity Score Interface

The methodology lists severity score as a final system output.

However, the methodology does not provide a complete specification for:

* Severity labels
* Number of severity classes
* Severity head architecture
* Severity loss
* Severity output dimension

Therefore this interface is:

```text
Severity score = TBD
```

The team must finalize this before implementing a severity prediction head.

Do not invent a severity architecture without team agreement.

---

# 29. Total Loss Interface

The methodology specifies:

```text
L_total =
    L_hate
    + lambda_1 * L_sarcasm
    + lambda_2 * L_contrastive
    + lambda_3 * L_target
```

where:

```text
lambda_1 = 0.3
lambda_2 = 0.1
lambda_3 = 0.2
```

## Components

```text
L_hate
    = BCE

L_sarcasm
    = BCE

L_contrastive
    = NT-Xent

L_target
    = multi-label BCE
```

The methodology specifies that the loss weights are tuned through grid search
on the validation set.

## Output

```text
total_loss: scalar
```

### Location

```text
src/multimodal_hate/training/
```

### Owner

Member 4

---

# 30. Training Interface

## Input

```text
Processed dataset
Model configuration
Training configuration
```

## Framework

```text
PyTorch
HuggingFace Transformers
```

## Training phases

### Phase 1 — Modality-Specific Pretraining

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

---

### Phase 2 — Fusion Module Training

Unfreeze:

```text
Top 4 transformer layers of each encoder
```

Train:

```text
Cross-attention fusion
Sarcasm gate
```

Dataset:

```text
Existing + scraped data
```

Settings:

```text
Learning rate = 2e-5
Warmup scheduler
Gradient clipping = 1.0
```

---

### Phase 3 — End-to-End Fine-Tuning + Adversarial Debiasing

Train:

```text
Full model
```

Add:

```text
Adversarial classifier
```

The adversarial classifier operates against protected attributes /
annotator-identity proxies.

Use:

```text
DANN-style adversarial loss
```

Final learning rate:

```text
5e-6
```

---

# 31. Training Infrastructure Interface

The methodology specifies:

```text
PyTorch
HuggingFace Transformers
Batch size = 32
Gradient accumulation = 4
Mixed precision = fp16
GPU = A100 / V100
Weights & Biases logging
Early stopping patience = 5
K-fold cross-validation = 5
```

These values belong in:

```text
configs/training.yaml
```

and should not be hard-coded inside model modules.

---

# 32. Evaluation Interface

## Input

```text
Ground-truth labels
Model predictions
Prediction probabilities
Explainability outputs
```

## Required metrics

```text
AUROC
Macro F1
Balanced Accuracy
False Positive Rate
Cohen's Kappa
GradCAM IoU
```

## Metric descriptions

### AUROC

Primary metric for the Hateful Memes benchmark.

### Macro F1

Used to account for class imbalance between hate and not-hate.

### Balanced Accuracy

Average of per-class recall/accuracy.

### False Positive Rate

Important for avoiding benign sarcastic samples being incorrectly flagged
as hate.

### Cohen's Kappa

Used for agreement analysis associated with sarcasm labelling.

### GradCAM IoU

Measures alignment between model attention maps and annotated hateful
regions.

### Location

```text
src/multimodal_hate/evaluation/
```

### Owner

Member 4

---

# 33. Explainability Interface

## Input

```text
Trained model
Meme image
Prediction
```

## Method

The methodology specifies:

```text
GradCAM
Spatial attention
Cross-modal attention
```

## Output

```text
Explainability map
```

The map should indicate visually relevant regions associated with the
prediction.

The exact implementation details of GradCAM extraction are:

```text
TBD
```

until the target ViT/fusion layer is selected.

### Location

```text
artifacts/explainability/
```

and implementation:

```text
src/multimodal_hate/evaluation/
```

### Owner

Member 4

---

# 34. Inference Interface

## Input

```text
Meme image
Optional post/title context
```

## Processing

```text
Image
   |
   v
OCR
   |
   v
Structured text
   |
   +------------------+
   |                  |
   v                  v
Image Encoders    Text Encoders
   |                  |
   +--------+---------+
            |
            v
      Sarcasm Module
            |
            v
      Sarcasm Gate
            |
            v
     Projection Layer
            |
            v
   Cross-Modal Fusion
            |
            v
    Classification Heads
            |
            v
       Predictions
```

## Output

Logical prediction object:

```python
{
    "hate_probability": float,
    "hate_label": int,
    "target_probabilities": list,
    "sarcasm_probability": float,
    "severity_score": float | None,
    "explainability": dict
}
```

Severity is optional until the severity head is formally specified.

### Location

```text
src/multimodal_hate/inference/
```

### Owner

Member 4

---

# 35. Ablation Interface

The methodology specifies the following ablation experiments.

## A. Without sarcasm module

```text
Baseline fusion only
```

## B. Without cross-attention

```text
Concatenation fusion
```

## C. Without contrastive loss

```text
Direct classification
```

## D. Text-only

```text
HateBERT
```

## E. Image-only

```text
ViT + hate head
```

## F. Without OCR text

```text
Image + post-title only
```

Each ablation must use the same evaluation protocol as the main model.

### Location

```text
experiments/ablations/
```

### Owner

Member 4

---

# 36. Public Function Interfaces

The following public function names are recommended for stable module
boundaries.

These names are implementation contracts and may be changed only by team
agreement.

## Member 1 — Data

```python
load_dataset(...)
```

```python
prepare_sample(...)
```

```python
run_ocr(...)
```

```python
build_dataset_record(...)
```

```python
create_data_splits(...)
```

---

## Member 2 — Encoders

```python
encode_image_vit(...)
```

```python
encode_image_clip(...)
```

```python
encode_text_hatebert(...)
```

```python
encode_text_clip(...)
```

```python
encode_text_roberta(...)
```

```python
extract_linguistic_features(...)
```

---

## Member 3 — Sarcasm

```python
predict_sarcasm(...)
```

```python
compute_incongruity(...)
```

```python
detect_sentiment_reversal(...)
```

```python
compute_sarcasm_gate(...)
```

---

## Member 4 — Fusion

```python
project_modalities(...)
```

```python
cross_modal_fusion(...)
```

```python
apply_sarcasm_gate(...)
```

```python
compute_total_loss(...)
```

```python
train_one_step(...)
```

```python
evaluate_model(...)
```

```python
run_inference(...)
```

---

# 37. Tensor Shape Contract

The following dimensions are explicitly supported by the methodology.

## Image

```text
ViT-L/14:
[B, N_img, 1024]
```

## Text

```text
HateBERT:
[B, N_txt, 768]
```

## CLIP image

```text
[B, 512]
```

## CLIP text

```text
[B, 512]
```

## Handcrafted linguistic features

```text
[B, 64]
```

## Projected image tokens

```text
[B, N_img, 512]
```

## Projected text tokens

```text
[B, N_txt, 512]
```

## Fused token sequence

```text
[B, N_fused, 512]
```

where:

```text
N_fused = N_img + N_txt
```

## Hate prediction

```text
[B, 2]
```

## Target-group prediction

```text
[B, 5]
```

## Sarcasm probability

```text
[B, 1]
```

## Important TBD dimensions

```text
N_img
N_txt
D_sarc
severity output dimension
final pooled representation implementation
supplementary visual feature dimension
```

These must not be silently changed.

---

# 38. Interface Rules for All Members

1. Never silently change a shared tensor dimension.
2. Never change another member's public function signature without agreement.
3. Every public module must document its input and output shapes.
4. Use batch-first tensor conventions.
5. Do not hard-code dataset paths.
6. Dataset paths belong in configuration files.
7. Model hyperparameters belong in configuration files.
8. API keys must never be committed.
9. Large datasets must not be committed directly to Git.
10. Model checkpoints must not be committed directly to Git unless explicitly
    approved and handled through the project's large-file strategy.
11. Add tests for every major module.
12. Use dummy tensors to test module interfaces before upstream modules are
    complete.
13. If an upstream module is incomplete, use a mock/stub with the agreed
    interface.
14. Do not modify another member's module without communicating the change.
15. Changes to this interface document require team agreement.
16. Any methodology parameter marked `TBD` must be resolved before it is
    required by another module.
17. Configuration values must not be duplicated across Python files.
18. Every experiment must record its configuration and results.
19. All experiments must be reproducible from a known configuration.
20. The `main` branch represents the stable integrated version.

---

# 39. Team Dependency Rules

## Member 1 -> Member 2

Member 1 provides:

```text
Image path
Structured OCR text
OCR regions
Post/title context
Dataset labels
Metadata
```

Member 2 must not depend on the internal implementation of Member 1's
scraper or OCR code.

Member 2 only depends on the standardized data record.

---

## Member 2 -> Member 3

Member 2 provides:

```text
CLIP image embedding
CLIP text embedding
Structured text representation
```

Member 3 uses these through documented interfaces.

Member 3 must not directly modify encoder internals.

---

## Member 2 + Member 3 -> Member 4

Member 2 provides:

```text
Projected image features
Projected text features
CLIP image embedding
CLIP text embedding
```

Member 3 provides:

```text
Sarcasm probability
Incongruity score
Sentiment reversal information
Sarcasm gate vector
```

Member 4 consumes these interfaces.

---

# 40. What Members Must NOT Modify

## Member 1 must NOT modify

```text
src/multimodal_hate/models/fusion/
src/multimodal_hate/training/
src/multimodal_hate/evaluation/
```

without team agreement.

## Member 2 must NOT modify

```text
src/multimodal_hate/models/sarcasm/
src/multimodal_hate/models/fusion/
```

without team agreement.

## Member 3 must NOT modify

```text
src/multimodal_hate/models/encoders/
src/multimodal_hate/models/fusion/
```

without team agreement.

## Member 4 must NOT modify

the internal data collection or encoder implementations without agreement.

Member 4 consumes their documented outputs.

---

# 41. TBD Decisions Requiring Team Agreement

The following items are not fully specified by the methodology and must
therefore be explicitly decided before implementation:

* Exact tokenizer maximum sequence lengths
* Exact image preprocessing resolution
* Exact `N_img`
* Exact `N_txt`
* Exact sarcasm gate dimension `d`
* Exact RoBERTa representation used downstream
* Exact supplementary visual-feature dimension
* Exact pooling operation after fusion
* Exact severity-score architecture
* Exact severity labels
* Exact severity loss
* Exact DANN loss formulation and weighting
* Exact protected/annotator proxy attributes
* Exact train/validation/test split for each dataset
* Exact mapping of heterogeneous dataset labels
* Exact missing-label handling
* Exact checkpoint naming/versioning convention

No member should independently invent one of these values.

Once decided, the value must be added to:

```text
docs/interfaces.md
```

and the relevant configuration file.

---

# 42. Configuration Ownership

Configuration values must live in:

```text
configs/
├── base.yaml
├── dataset.yaml
├── model.yaml
├── training.yaml
├── evaluation.yaml
└── experiments/
```

Examples:

```text
dataset.yaml
    dataset paths
    split settings
    preprocessing

model.yaml
    encoder names
    fusion dimension
    attention layers
    attention heads
    dropout

training.yaml
    learning rates
    batch size
    epochs
    loss weights
    scheduler
    gradient clipping

evaluation.yaml
    metrics
    cross-validation
    explainability settings
```

---

# 43. Final Interface Summary

```text
                  MEMBER 1
        DATA + OCR + PREPROCESSING
                    |
                    | standardized sample
                    v
                  MEMBER 2
          IMAGE / TEXT ENCODERS
                    |
       +------------+------------+
       |                         |
       | embeddings              | features
       v                         v
                  MEMBER 3
             SARCASM MODULE
                    |
                    | sarcasm gate
                    v
                  MEMBER 4
       PROJECTION + CROSS-ATTENTION
                    |
                    v
             SELF-ATTENTION
                    |
                    v
          SARCASM-CONDITIONED
             FUSED FEATURES
                    |
          +---------+---------+
          |         |         |
          v         v         v
        HATE      TARGET    SARCASM
        HEAD       HEAD      HEAD
          |         |         |
          +---------+---------+
                    |
                    v
       PREDICTIONS + EXPLAINABILITY
```

---

# 44. Definition of Interface Stability

An interface is considered stable when:

* Input field names are agreed.
* Output field names are agreed.
* Tensor dimensions are documented.
* Data types are documented.
* Public function names are documented.
* Unit tests exist.
* A dummy input successfully passes through the module.
* Downstream members can use the output without accessing internal code.

Until these conditions are met, the interface should be considered:

```text
IN DEVELOPMENT
```

After team approval:

```text
STABLE
```

---

# 45. Source-of-Truth Rule

The project methodology document is the primary research source.

This interface document translates the methodology into software contracts.

If a conflict exists:

```text
Project Methodology
        >
docs/interfaces.md
        >
Implementation
```

Any implementation change that intentionally deviates from the methodology
must be documented and approved by the team.

````

This version is safer than the earlier one because it **doesn't pretend the methodology specifies details that it doesn't**. For example, the source explicitly specifies the 512-dimensional shared fusion space, four co-attention layers, eight heads, and 0.1 dropout, so those are fixed here. :contentReference[oaicite:3]{index=3} It also explicitly specifies the sarcasm components and their outputs, including SarcasmBERT, CLIP incongruity, VADER+BERT sentiment reversal, and the sigmoid gate. :contentReference[oaicite:4]{index=4}

The training phases, learning rates, infrastructure, and evaluation metrics are also preserved from the methodology rather than invented. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

### What to do now

1. Open `docs/interfaces.md` → **Edit**.
2. **Delete the current 31 lines completely.**
3. Paste the entire block above.
4. Click **Preview** and check that the headings/tables/code blocks render properly.
5. **Don't commit yet** if anything looks wrong.
6. If it renders correctly, commit with:

```text
Complete module interface contract
````
