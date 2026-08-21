import pytest

from src.multimodal_hate.data.processing.seeding import (
    TARGET_GROUPS,
    build_hashtag_seeds,
    build_seed_terms,
    expand_with_embeddings,
    generate_hashtags,
    normalize_term,
    normalize_terms,
)


def test_normalize_term():
    assert normalize_term("#Religion") == "religion"
    assert normalize_term("  Gender  ") == "gender"


def test_normalize_terms_removes_duplicates():
    result = normalize_terms(
        ["Race", "#Race", " religion ", ""]
    )

    assert result == ["race", "religion"]


def test_build_seed_terms():
    result = build_seed_terms(
        base_terms={
            "race": ["race", "#ethnicity"],
            "religion": ["religion"],
            "gender": [],
            "sexuality": [],
        },
        custom_expansion={
            "race": ["ethnic group"],
        },
    )

    assert result["race"] == [
        "ethnicity",
        "ethnic group",
        "race",
    ]

    assert result["religion"] == ["religion"]


def test_build_seed_terms_with_lexicon():
    result = build_seed_terms(
        base_terms={
            "race": ["race"],
            "religion": ["religion"],
            "gender": ["gender"],
            "sexuality": ["sexuality"],
        },
        lexicon_terms=["term_one", "term_two"],
    )

    for group in TARGET_GROUPS:
        assert "term_one" in result[group]
        assert "term_two" in result[group]


def test_generate_hashtags():
    result = generate_hashtags(
        {
            "race": ["race", "ethnicity"],
            "religion": ["religion"],
        }
    )

    assert result["race"] == [
        "#race",
        "#ethnicity",
    ]

    assert result["religion"] == ["#religion"]


def test_build_hashtag_seeds():
    result = build_hashtag_seeds(
        base_terms={
            "race": ["race"],
            "religion": ["religion"],
            "gender": ["gender"],
            "sexuality": ["sexuality"],
        }
    )

    assert "keywords" in result["race"]
    assert "hashtags" in result["race"]

    assert "#race" in result["race"]["hashtags"]


def test_embedding_expansion():
    numpy = pytest.importorskip("numpy")

    vectors = numpy.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )

    vocabulary = [
        "race",
        "ethnicity",
        "religion",
    ]

    result = expand_with_embeddings(
        terms=["race"],
        embedding_vectors=vectors,
        vocabulary=vocabulary,
        top_k=1,
        similarity_threshold=0.5,
    )

    assert "race" in result
    assert "ethnicity" in result


def test_embedding_expansion_invalid_top_k():
    with pytest.raises(ValueError):
        expand_with_embeddings(
            terms=["race"],
            embedding_vectors=[],
            vocabulary=[],
            top_k=-1,
        )