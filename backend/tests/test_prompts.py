import itertools
from server.prompts import prompt_stream


def test_yields_only_vocab():
    vocab = ["a", "b", "c"]
    got = list(itertools.islice(prompt_stream(vocab, seed=1), 50))
    assert set(got) <= set(vocab)
    assert len(got) == 50  # infinite-capable


def test_no_immediate_repeats():
    got = list(itertools.islice(prompt_stream(["a", "b", "c"], seed=2), 100))
    assert all(a != b for a, b in zip(got, got[1:]))


def test_seed_is_deterministic():
    a = list(itertools.islice(prompt_stream(["a", "b", "c", "d"], seed=7), 20))
    b = list(itertools.islice(prompt_stream(["a", "b", "c", "d"], seed=7), 20))
    assert a == b
