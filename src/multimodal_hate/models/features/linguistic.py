import re
from typing import List

import torch


FEATURE_DIM = 64


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _words(text: str) -> List[str]:
    """Return lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# Slur features
# Indices: 0-7
# ---------------------------------------------------------------------------

def _slur_features(text: str) -> List[float]:
    """
    Detect basic hate/slur-related lexical indicators.

    These are deterministic placeholder lexical indicators.
    The project can later extend this using the HateXplain lexicon.
    """

    slur_terms = {
        "racist",
        "racism",
        "hate",
        "hateful",
        "bigot",
        "bigoted",
        "homophobic",
        "sexist",
    }

    words = _words(text)

    return [
        float(term in words)
        for term in sorted(slur_terms)
    ]


# ---------------------------------------------------------------------------
# Negation features
# Indices: 8-15
# ---------------------------------------------------------------------------

def _negation_features(text: str) -> List[float]:
    """
    Extract negation features.

    Index 8:
        Number of negation words.

    Index 10:
        Binary indicator showing whether negation is present.
    """

    negations = {
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "nothing",
        "nobody",
        "nowhere",
        "without",
    }

    words = _words(text)

    count = sum(word in negations for word in words)

    features = [0.0] * 8

    # Required project/test interface
    features[0] = float(count)          # global index 8
    features[2] = float(count > 0)      # global index 10

    # Additional useful negation statistics
    features[1] = float(min(count, 3))
    features[3] = float(count >= 2)
    features[4] = float(count >= 3)

    return features


# ---------------------------------------------------------------------------
# Intensifier features
# Indices: 16-23
# ---------------------------------------------------------------------------

def _intensifier_features(text: str) -> List[float]:
    """
    Extract intensifier features.

    Index 16:
        Number of intensifier words.

    Index 18:
        Binary indicator showing whether an intensifier is present.
    """

    intensifiers = {
        "very",
        "extremely",
        "really",
        "so",
        "too",
        "highly",
        "completely",
        "totally",
        "absolutely",
        "literally",
    }

    words = _words(text)

    count = sum(word in intensifiers for word in words)

    features = [0.0] * 8

    # Required project/test interface
    features[0] = float(count)          # global index 16
    features[2] = float(count > 0)      # global index 18

    # Additional intensifier statistics
    features[1] = float(min(count, 3))
    features[3] = float(count >= 2)
    features[4] = float(count >= 3)

    return features


# ---------------------------------------------------------------------------
# Sentiment features
# Indices: 24-31
# ---------------------------------------------------------------------------

def _sentiment_features(text: str) -> List[float]:
    """
    Extract lexical sentiment features.

    Index 24:
        Positive word count.

    Index 25:
        Negative word count.
    """

    positive_words = {
        "good",
        "great",
        "love",
        "excellent",
        "happy",
        "wonderful",
        "best",
        "nice",
        "positive",
        "amazing",
    }

    negative_words = {
        "bad",
        "hate",
        "awful",
        "terrible",
        "worst",
        "angry",
        "sad",
        "horrible",
        "negative",
        "disgusting",
    }

    words = _words(text)

    positive_count = sum(word in positive_words for word in words)
    negative_count = sum(word in negative_words for word in words)

    features = [0.0] * 8

    # Required project/test interface
    features[0] = float(positive_count)  # global index 24
    features[1] = float(negative_count)  # global index 25

    # Additional sentiment indicators
    features[2] = float(positive_count > 0)
    features[3] = float(negative_count > 0)
    features[4] = float(positive_count > negative_count)
    features[5] = float(negative_count > positive_count)

    return features


# ---------------------------------------------------------------------------
# Punctuation features
# Indices: 32-39
# ---------------------------------------------------------------------------

def _punctuation_features(text: str) -> List[float]:
    """
    Extract punctuation features.

    Index 32:
        Number of exclamation marks.

    Index 33:
        Number of question marks.
    """

    exclamation_count = text.count("!")
    question_count = text.count("?")
    punctuation_count = len(re.findall(r"[^\w\s]", text))

    features = [0.0] * 8

    # Required project/test interface
    features[0] = float(exclamation_count)  # global index 32
    features[1] = float(question_count)     # global index 33

    # Additional punctuation statistics
    features[2] = float(punctuation_count)
    features[3] = float(exclamation_count > 0)
    features[4] = float(question_count > 0)

    if text:
        features[5] = exclamation_count / len(text)
        features[6] = question_count / len(text)
        features[7] = punctuation_count / len(text)

    return features


# ---------------------------------------------------------------------------
# Text statistics
# Indices: 40-47
# ---------------------------------------------------------------------------

def _text_statistics(text: str) -> List[float]:
    """Extract general text statistics."""

    words = re.findall(r"\b\w+\b", text)

    characters = len(text)
    word_count = len(words)

    uppercase_count = sum(char.isupper() for char in text)
    digit_count = sum(char.isdigit() for char in text)

    return [
        float(word_count),
        float(characters),
        float(uppercase_count),
        float(digit_count),
        float(len(text.strip()) == 0),
        float(len(set(words))),
        float(sum(len(word) for word in words) / max(word_count, 1)),
        float(max((len(word) for word in words), default=0)),
    ]


# ---------------------------------------------------------------------------
# Target-group keyword features
# Indices: 48-63
# ---------------------------------------------------------------------------

def _keyword_features(text: str) -> List[float]:
    """
    Detect target-group and hate-related keywords.

    The output is exactly 16 dimensions.
    """

    keywords = {
        "race",
        "ethnicity",
        "religion",
        "gender",
        "woman",
        "women",
        "man",
        "men",
        "gay",
        "lesbian",
        "trans",
        "disabled",
        "disability",
        "immigrant",
        "foreigner",
        "minority",
    }

    words = set(_words(text))

    return [
        float(keyword in words)
        for keyword in sorted(keywords)
    ]


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

def extract_linguistic_features(texts) -> torch.Tensor:
    """
    Extract handcrafted linguistic features.

    Parameters
    ----------
    texts:
        A single string or a list of strings.

    Returns
    -------
    torch.Tensor
        Tensor with shape [B, 64].

    Feature layout
    --------------
    0-7:
        Slur-related lexical indicators

    8-15:
        Negation features

    16-23:
        Intensifier features

    24-31:
        Sentiment features

    32-39:
        Punctuation features

    40-47:
        Text statistics

    48-63:
        Target-group keyword features
    """

    if isinstance(texts, str):
        texts = [texts]

    features = []

    for text in texts:
        if text is None:
            text = ""

        text = str(text)

        vector = []

        vector.extend(_slur_features(text))        # 0-7
        vector.extend(_negation_features(text))    # 8-15
        vector.extend(_intensifier_features(text)) # 16-23
        vector.extend(_sentiment_features(text))   # 24-31
        vector.extend(_punctuation_features(text)) # 32-39
        vector.extend(_text_statistics(text))      # 40-47
        vector.extend(_keyword_features(text))     # 48-63

        # Safety check: interface must always be [B, 64]
        if len(vector) < FEATURE_DIM:
            vector.extend([0.0] * (FEATURE_DIM - len(vector)))
        elif len(vector) > FEATURE_DIM:
            vector = vector[:FEATURE_DIM]

        features.append(vector)

    return torch.tensor(features, dtype=torch.float32)