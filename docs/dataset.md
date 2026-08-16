````markdown
# Dataset Documentation

## Multimodal Hate Speech Detection with Sarcasm-Aware Transformer Fusion

This document describes the datasets, data collection, preprocessing, annotation, filtering, balancing, and augmentation methodology used by the project.

The project methodology is the primary source of truth.

---

# 1. Dataset Overview

The project combines existing multimodal datasets with newly collected data.

The dataset sources are:

1. Hateful Memes Challenge
2. MultiOFF
3. iSarcasm
4. MUStARD
5. HateSPAN
6. MemeCap
7. Reddit
8. Twitter/X
9. Know Your Meme metadata

The resulting dataset is intended to support multimodal learning of:

- Hate speech
- Sarcasm / irony
- Visual-text incongruity
- Target-group information

---

# 2. Existing Datasets

## 2.1 Hateful Memes Challenge

The Hateful Memes Challenge dataset is the primary multimodal hate-speech benchmark.

The methodology specifies approximately:

```text
~10,000 memes
````

with:

* Binary hate labels
* Confounder sets

The confounders are intended to reduce unimodal shortcuts and encourage genuine multimodal reasoning.

---

## 2.2 MultiOFF

MultiOFF is a multimodal offensive-content dataset.

The methodology specifies approximately:

```text
~750 memes
```

It contains:

* Offensive / not-offensive labels
* Image information
* Overlaid-text annotations

This dataset provides additional multimodal offensive-content supervision.

---

## 2.3 iSarcasm

iSarcasm provides sarcasm supervision.

It is used for training the sarcasm-related components of the system.

The project methodology combines iSarcasm with other sarcasm resources to provide sarcasm-related supervision.

---

## 2.4 MUStARD

MUStARD provides multimodal sarcasm information.

The methodology specifies adapting the relevant:

```text
text + image
```

information for this project.

The original dataset modality should not be silently converted into a different dataset format during implementation.

---

## 2.5 HateSPAN

HateSPAN provides:

```text
span-level hateful-token annotations
```

These annotations are useful for identifying text regions associated with hateful content.

---

## 2.6 MemeCap

MemeCap provides:

```text
rich captions for meme images
```

These captions are used for semantic grounding of visual content.

---

# 3. Newly Collected Data

The project methodology specifies additional data collection from:

```text
Reddit
Twitter/X
Know Your Meme
```

These sources are intended to increase dataset diversity and improve cross-platform generalization.

---

# 4. Reddit Data Collection

Reddit collection uses:

```text
PRAW
```

The methodology specifies the following target subreddits:

```text
r/dankmemes
r/PoliticalHumor
r/ComedyCemetery
r/ControversialHumor
```

The collection strategy is:

```text
Reddit posts
      ↓
Select top-all posts
      ↓
Filter image posts
      ↓
Extract image
      ↓
Extract post title
      ↓
Use title as meme-caption context
```

Estimated collection size:

```text
15K–25K samples
```

The Reddit sample contains:

```text
Image
+
Post title
```

The post title is used as contextual information for the meme.

---

# 5. Twitter/X Data Collection

The methodology specifies using the Twitter Academic API methodology.

The collection focuses on:

* Hate-associated hashtags
* Target-group keywords
* Relevant hate-associated terms

An example specified hashtag is:

```text
#hateful
```

Target-group slurs are obtained using the project keyword lists.

Estimated collection size:

```text
~10K tweets
```

The collected information is used as additional multimodal hate-related data.

---

# 6. Know Your Meme Metadata

Know Your Meme is used to obtain meme-template information.

The methodology specifies scraping:

```text
Meme template descriptions
```

These descriptions are used as metadata for:

```text
Visual grounding
+
Meme semantic context
```

KYM metadata is combined with OCR and other contextual information where appropriate.

---

# 7. Keyword and Hashtag Seeding

The scraping process begins by defining seed terms for target groups.

The methodology specifies target groups including:

```text
Race
Religion
Gender
Sexuality
```

The seed vocabulary uses:

```text
HateXplain slur lexicons
+
Custom expansion
```

Custom expansion uses:

```text
WordNet
+
Embedding nearest neighbors
```

The purpose is to improve coverage of relevant terminology.

---

# 8. Image Filtering

Only relevant image posts are retained.

The filtering pipeline is:

```text
Collected Posts
      ↓
Image-post filtering
      ↓
Perceptual hash deduplication
      ↓
Meme-template filtering
      ↓
Accepted meme images
```

---

# 9. Perceptual Hash Deduplication

Perceptual hashing is used to identify visually duplicated or highly similar images.

The methodology specifies:

```text
pHash similarity threshold ≥ 95%
```

Images meeting the duplicate-similarity criterion are removed to reduce duplicate samples.

---

# 10. Meme Template Filtering

Non-meme images are filtered using a meme-template classifier.

The methodology specifies:

```text
ResNet-50
```

fine-tuned on:

```text
Know Your Meme
```

data.

The classifier is used to distinguish relevant meme images from non-meme images.

---

# 11. OCR Processing

OCR is a major part of the dataset preparation pipeline.

The methodology specifies an OCR ensemble:

```text
EasyOCR
+
TrOCR
```

The OCR system extracts text embedded inside meme images.

---

# 12. OCR Metadata

The OCR output stores:

```text
OCR text
+
Bounding boxes
+
Font region
+
Confidence scores
```

The methodology specifies positional regions:

```text
TOP
BOTTOM
CAPTION
```

This information is retained instead of flattening all OCR text into a single unstructured string.

---

# 13. Structured OCR Context

OCR text is merged with the post title where available.

The resulting representation is structured contextual information:

```text
[TOP] ...
[BOTTOM] ...
[CAPTION] ...

Post Title: ...
```

The exact serialization format should follow the module interface contract.

---

# 14. Annotation Pipeline

The newly collected data requires annotation.

The methodology specifies:

```text
3 annotators
```

using:

```text
MTurk
+
Label Studio
```

The annotation strategy uses majority voting.

---

# 15. Annotation Labels

Each relevant sample is annotated for the following information.

## 15.1 Hate Label

Binary label:

```text
Hate
Not-Hate
```

---

## 15.2 Target Group

The methodology specifies target groups including:

```text
Race
Religion
Gender
Sexuality
```

The exact target-group label mapping must remain consistent with the project configuration.

---

## 15.3 Sarcasm / Irony

Each sample receives a:

```text
Sarcasm / Irony flag
```

This enables joint hate + sarcasm learning.

---

## 15.4 Incongruity Type

The annotation includes the type of visual-text incongruity.

The methodology specifies:

```text
Visual-text conflict
Hyperbole
Exaggeration
```

---

# 16. Annotation Agreement

The project targets:

```text
Cohen's κ ≥ 0.70
```

as the inter-annotator agreement threshold.

If the agreement threshold is not achieved, the annotation process should be reviewed before the resulting labels are treated as final project labels.

---

# 17. Dual Annotation

A major feature of the dataset is simultaneous annotation of:

```text
Hate
+
Sarcasm
```

for the same meme.

This enables the model to learn the joint relationship:

```text
P(hate, sarcasm | meme)
```

rather than treating hate and sarcasm as completely separate datasets.

---

# 18. Class Balancing

The methodology specifies:

```text
SMOTE
```

for minority-class balancing.

SMOTE is applied to:

```text
Feature embeddings
```

rather than directly generating new raw meme images.

The exact point at which SMOTE is applied must remain consistent with the training/data pipeline.

---

# 19. Image Augmentation

The methodology specifies three image augmentation strategies:

### Horizontal Flip

Images may be horizontally flipped.

### Colour Jitter

Colour properties are randomly modified.

### Text Region Masking

Text regions are masked using a:

```text
SimCLR-style
```

augmentation approach.

The purpose is to improve robustness to variations in visual presentation.

---

# 20. Text Augmentation

The methodology specifies back-translation:

```text
English
   ↓
German
   ↓
English
```

The specified translation model is:

```text
MarianMT
```

The purpose is to create textual variation while preserving the underlying semantic content.

---

# 21. Dataset Processing Pipeline

The complete dataset preparation flow is:

```text
Existing Datasets
       +
Reddit
       +
Twitter/X
       +
Know Your Meme
       ↓
Data Collection
       ↓
Keyword / Hashtag Filtering
       ↓
Image Filtering
       ↓
pHash Deduplication
       ↓
Meme Template Filtering
       ↓
OCR
       ↓
Structured OCR
       ↓
Dual Annotation
       ↓
Majority Voting
       ↓
Agreement Check
       ↓
Class Balancing
       ↓
Image Augmentation
       +
Text Augmentation
       ↓
Processed Dataset
```

---

# 22. Dataset Sample Representation

A processed sample should conceptually contain:

```text
{
    image,
    text,
    ocr_text,
    metadata,
    hate_label,
    target_group,
    sarcasm_label,
    incongruity_type
}
```

This is a conceptual representation of the information required by the downstream pipeline.

The exact serialized schema must follow the project's module interface contract.

---

# 23. Data Directory Mapping

The repository uses:

```text
data/
├── raw/
├── interim/
├── processed/
└── splits/
```

## raw/

Original downloaded or collected data.

Examples:

```text
raw images
raw metadata
original annotations
```

Raw data should not be modified after collection.

---

## interim/

Intermediate data produced during preprocessing.

Examples:

```text
OCR outputs
deduplication results
temporary metadata
filtered samples
```

---

## processed/

Final model-ready dataset representations.

Examples:

```text
processed metadata
processed labels
processed OCR
processed feature-ready records
```

---

## splits/

Dataset split information.

Examples:

```text
train
validation
test
```

Only split definitions and metadata should be stored here when the actual dataset is too large to commit.

---

# 24. Dataset Configuration

Dataset-related configuration is maintained in:

```text
configs/dataset.yaml
```

This configuration should contain dataset paths and dataset-processing parameters rather than hard-coded paths inside Python modules.

---

# 25. Data Privacy and Repository Rules

Raw datasets may contain copyrighted, platform-derived, or otherwise restricted material.

Therefore:

```text
Do not automatically commit raw datasets to GitHub.
```

The repository should contain:

* Dataset documentation
* Download instructions
* Dataset metadata where permitted
* Preprocessing scripts
* Split definitions where permitted

Actual restricted datasets should remain outside the repository unless redistribution is explicitly permitted.

---

# 26. Reproducibility

Every dataset preparation run should record:

```text
Dataset source
Collection date
Filtering parameters
Deduplication threshold
OCR configuration
Annotation version
Augmentation configuration
Dataset version
Train/validation/test split information
```

This allows another team member to reproduce the same processed dataset.

---

# 27. Dataset Source-of-Truth Rule

The project methodology is the primary source of truth for dataset construction.

The implementation must not silently:

* Replace a specified dataset
* Change annotation labels
* Change filtering thresholds
* Remove required preprocessing
* Add new labels
* Change the augmentation strategy

Any intentional deviation must be documented and approved by the team.

---

# 28. Important TBD Items

The methodology does not specify every operational detail required for implementation.

The following should remain configurable or explicitly agreed upon before implementation:

```text
Exact final combined dataset size
Exact train/validation/test split ratios
Exact file serialization format
Exact dataset directory names for each external source
Exact SMOTE implementation parameters
Exact augmentation probabilities
Exact OCR confidence filtering rule
Exact handling of missing modalities
Exact target-group label encoding
```

These values should not be silently invented.

---

# 29. Dataset-to-Model Interface

The final processed dataset feeds the model pipeline as:

```text
Processed Image
      +
Processed OCR/Text
      +
Metadata
      +
Labels
      ↓
Image Encoder
+
Text Encoder
+
Sarcasm Module
      ↓
Fusion
      ↓
Classification
```

The exact tensor-level interfaces are defined separately in:

```text
docs/interfaces.md
```

---

# 30. Summary

The dataset methodology combines established multimodal hate/sarcasm resources with newly collected Reddit and Twitter/X data and Know Your Meme metadata.

The key processing stages are:

```text
Collection
   ↓
Filtering
   ↓
Deduplication
   ↓
OCR
   ↓
Annotation
   ↓
Balancing
   ↓
Augmentation
   ↓
Processed Multimodal Dataset
```

The resulting dataset provides the multimodal information required by the project's sarcasm-aware transformer fusion architecture.

````
