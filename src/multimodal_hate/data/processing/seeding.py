from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set


TARGET_GROUPS = (
    "race",
    "religion",
    "gender",
    "sexuality",
)


def normalize_term(term: str) -> str:
    """
    Normalize a keyword or hashtag seed.

    Hashtags are converted to plain lowercase terms while
    preserving the semantic content of the seed.
    """

    if not isinstance(term, str):
        raise TypeError("Seed term must be a string.")

    term = term.strip().lower()

    if term.startswith("#"):
        term = term[1:]

    return term.strip()


def normalize_terms(terms: Iterable[str]) -> List[str]:
    """
    Normalize and deduplicate a collection of seed terms.
    """

    normalized: Set[str] = set()

    for term in terms:
        value = normalize_term(term)

        if value:
            normalized.add(value)

    return sorted(normalized)


def load_lexicon(path: str) -> List[str]:
    """
    Load a HateXplain-style lexicon from a text file.

    One term is expected per line.

    Empty lines and comment lines beginning with '#'
    are ignored.
    """

    lexicon_path = Path(path)

    if not lexicon_path.exists():
        raise FileNotFoundError(
            f"Lexicon file not found: {lexicon_path}"
        )

    terms: List[str] = []

    with lexicon_path.open("r", encoding="utf-8") as file:
        for line in file:
            term = line.strip()

            if not term or term.startswith("#"):
                continue

            terms.append(term)

    return normalize_terms(terms)


def build_seed_terms(
    base_terms: Mapping[str, Sequence[str]],
    lexicon_terms: Optional[Iterable[str]] = None,
    custom_expansion: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Build keyword seeds for the target groups.

    Parameters
    ----------
    base_terms:
        Initial keywords grouped by target group.

    lexicon_terms:
        Terms obtained from the HateXplain slur lexicon.

    custom_expansion:
        Additional manually defined terms grouped by target group.

    Returns
    -------
    Dict[str, List[str]]
        Normalized and deduplicated terms for each target group.
    """

    result: Dict[str, List[str]] = {
        group: [] for group in TARGET_GROUPS
    }

    for group in TARGET_GROUPS:
        base = list(base_terms.get(group, []))

        custom = []
        if custom_expansion is not None:
            custom = list(custom_expansion.get(group, []))

        lexicon = list(lexicon_terms) if lexicon_terms is not None else []

        # Preserve the intended seed priority:
        # 1. Hashtag-derived base terms
        # 2. Custom expansion terms
        # 3. Remaining base terms
        # 4. HateXplain lexicon terms
        #
        # This also preserves insertion order while removing duplicates.

        hashtag_terms: List[str] = []
        normal_base_terms: List[str] = []

        for term in base:
            if isinstance(term, str) and term.strip().startswith("#"):
                hashtag_terms.append(term)
            else:
                normal_base_terms.append(term)

        combined = (
            hashtag_terms
            + custom
            + normal_base_terms
            + lexicon
        )

        normalized_terms: List[str] = []
        seen: Set[str] = set()

        for term in combined:
            normalized = normalize_term(term)

            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_terms.append(normalized)

        result[group] = normalized_terms

    return result


def generate_hashtags(
    keywords: Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """
    Convert keyword seeds into hashtag seeds.

    Example
    -------
    "example" -> "#example"
    """

    hashtags: Dict[str, List[str]] = {}

    for group, terms in keywords.items():
        hashtags[group] = [
            f"#{normalize_term(term)}"
            for term in terms
            if normalize_term(term)
        ]

    return hashtags


def expand_with_wordnet(
    terms: Iterable[str],
    max_synonyms_per_term: int = 10,
) -> List[str]:
    """
    Expand seed terms using WordNet synonyms.

    WordNet is optional. If the NLTK WordNet corpus is not
    installed, a clear LookupError is raised.
    """

    if max_synonyms_per_term < 0:
        raise ValueError(
            "max_synonyms_per_term must be non-negative."
        )

    try:
        from nltk.corpus import wordnet
    except ImportError as exc:
        raise ImportError(
            "NLTK is required for WordNet expansion."
        ) from exc

    expanded: Set[str] = set(normalize_terms(terms))

    for term in terms:
        normalized = normalize_term(term)

        if not normalized:
            continue

        synsets = wordnet.synsets(normalized)

        count = 0

        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = normalize_term(
                    lemma.name().replace("_", " ")
                )

                if synonym:
                    expanded.add(synonym)

                count += 1

                if count >= max_synonyms_per_term:
                    break

            if count >= max_synonyms_per_term:
                break

    return sorted(expanded)


def expand_with_embeddings(
    terms: Sequence[str],
    embedding_vectors,
    vocabulary: Sequence[str],
    top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> List[str]:
    """
    Expand seed terms using nearest neighbors from
    pre-computed word embeddings.

    Parameters
    ----------
    terms:
        Existing seed terms.

    embedding_vectors:
        Matrix containing one embedding vector per vocabulary item.

    vocabulary:
        Vocabulary corresponding to embedding_vectors.

    top_k:
        Maximum number of nearest neighbors per seed.

    similarity_threshold:
        Minimum cosine similarity required for a neighbor.

    Notes
    -----
    This function does not download or train an embedding model.
    The caller supplies the embeddings, keeping this module
    independent of a specific embedding implementation.
    """

    if top_k < 0:
        raise ValueError("top_k must be non-negative.")

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between 0.0 and 1.0."
        )

    if len(vocabulary) != len(embedding_vectors):
        raise ValueError(
            "Vocabulary and embedding vectors must have "
            "the same number of entries."
        )

    if top_k == 0 or not terms:
        return sorted(set(normalize_terms(terms)))

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "NumPy is required for embedding expansion."
        ) from exc

    vectors = np.asarray(embedding_vectors, dtype=float)

    if vectors.ndim != 2:
        raise ValueError(
            "embedding_vectors must be a 2-dimensional matrix."
        )

    vocabulary_normalized = [
        normalize_term(word)
        for word in vocabulary
    ]

    vocabulary_index = {
        word: index
        for index, word in enumerate(vocabulary_normalized)
        if word
    }

    norms = np.linalg.norm(vectors, axis=1)

    expanded: Set[str] = set(normalize_terms(terms))

    for term in normalize_terms(terms):
        if term not in vocabulary_index:
            continue

        index = vocabulary_index[term]

        if norms[index] == 0:
            continue

        similarities = (
            vectors @ vectors[index]
        ) / (
            norms * norms[index]
        )

        ranked_indices = np.argsort(-similarities)

        added = 0

        for neighbor_index in ranked_indices:
            neighbor = vocabulary_normalized[neighbor_index]

            if not neighbor or neighbor == term:
                continue

            similarity = float(similarities[neighbor_index])

            if similarity < similarity_threshold:
                continue

            expanded.add(neighbor)

            added += 1

            if added >= top_k:
                break

    return sorted(expanded)


def build_hashtag_seeds(
    base_terms: Mapping[str, Sequence[str]],
    lexicon_terms: Optional[Iterable[str]] = None,
    custom_expansion: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build both keyword and hashtag seeds.

    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        Example structure:

        {
            "race": {
                "keywords": [...],
                "hashtags": [...]
            },
            ...
        }
    """

    keywords = build_seed_terms(
        base_terms=base_terms,
        lexicon_terms=lexicon_terms,
        custom_expansion=custom_expansion,
    )

    hashtags = generate_hashtags(keywords)

    return {
        group: {
            "keywords": keywords[group],
            "hashtags": hashtags[group],
        }
        for group in TARGET_GROUPS
    }