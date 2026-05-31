import random
from typing import Iterator


def prompt_stream(vocab: list[str], seed: int = 0) -> Iterator[str]:
    """Yield an endless stream of glosses from `vocab`, never repeating twice
    in a row (when the vocab has > 1 entry). Deterministic for a given seed."""
    if not vocab:
        raise ValueError("vocab must be non-empty")
    rng = random.Random(seed)
    prev = None
    while True:
        choice = rng.choice(vocab)
        if choice == prev and len(vocab) > 1:
            continue
        prev = choice
        yield choice
