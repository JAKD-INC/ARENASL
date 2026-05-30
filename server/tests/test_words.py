"""Deterministic word stream + word_strength lookup."""

from __future__ import annotations

import pytest

from app import words


def test_word_at_is_deterministic(domain):
    seq_a = [words.word_at(12345, i).word for i in range(50)]
    seq_b = [words.word_at(12345, i).word for i in range(50)]
    assert seq_a == seq_b  # same seed + index -> identical word


def test_different_seeds_diverge(domain):
    a = [words.word_at(1, i).word for i in range(50)]
    b = [words.word_at(2, i).word for i in range(50)]
    assert a != b


def test_stream_uses_more_than_one_word(domain):
    distinct = {words.word_at(999, i).word for i in range(100)}
    assert len(distinct) > 1


def test_word_strength_is_dataset_difficulty(domain):
    ds = words.get_dataset()
    entry = ds.entries[0]
    assert ds.word_strength(entry.word) == entry.difficulty


def test_word_strength_unknown_raises(domain):
    with pytest.raises(KeyError):
        words.get_dataset().word_strength("definitely-not-a-word")


def test_random_seed_is_in_range(domain):
    s = words.random_seed()
    assert 0 <= s < 2**31
