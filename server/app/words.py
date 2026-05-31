"""Sign dataset loader (the `word_strength` source) and word-stream generator.

The dataset maps every sign to a word and a `difficulty` score — that difficulty
*is* the word strength. It's static reference data, loaded into memory once at
startup (fail fast if missing/unparseable) and shared with clients via GET /signs
so client and server use the identical ordered set.

Phase 1a implements loading + lookup. The seeded PRNG word stream lands in 1e.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

_MASK64 = (1 << 64) - 1
DEFAULT_DIFFICULTY = 2.0  # word_strength for a templated gloss missing from the catalog


@dataclass(frozen=True)
class SignEntry:
    sign_id: str
    word: str
    difficulty: float  # == word_strength


@dataclass(frozen=True)
class SignDataset:
    version: str
    entries: tuple[SignEntry, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("sign dataset has no entries")

    def to_payload(self) -> dict:
        """Shape served by GET /signs; clients must hold this exact version."""
        return {
            "version": self.version,
            "entries": [
                {"sign_id": e.sign_id, "word": e.word, "difficulty": e.difficulty}
                for e in self.entries
            ],
        }

    def word_strength(self, word: str) -> float:
        for e in self.entries:
            if e.word == word:
                return e.difficulty
        raise KeyError(f"word not in dataset: {word!r}")


def load_dataset(path: str | Path) -> SignDataset:
    """Load and validate the dataset file. Raises on any problem (fail fast)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sign dataset not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"sign dataset is not valid JSON: {p} ({exc})") from exc

    version = raw.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("sign dataset missing a string 'version'")

    entries: list[SignEntry] = []
    seen_words: set[str] = set()
    for i, item in enumerate(raw.get("entries", [])):
        try:
            entry = SignEntry(
                sign_id=str(item["sign_id"]),
                word=str(item["word"]),
                difficulty=float(item["difficulty"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"sign dataset entry {i} is malformed: {exc}") from exc
        if entry.word in seen_words:
            raise ValueError(f"sign dataset has a duplicate word: {entry.word!r}")
        seen_words.add(entry.word)
        entries.append(entry)

    return SignDataset(version=version, entries=tuple(entries))


# --- module-level singleton, set at startup ---------------------------------

_dataset: SignDataset | None = None


def init_dataset(path: str | Path) -> SignDataset:
    global _dataset
    _dataset = load_dataset(path)
    return _dataset


def get_dataset() -> SignDataset:
    if _dataset is None:
        raise RuntimeError("sign dataset not loaded; call init_dataset() at startup")
    return _dataset


def restrict_to(glosses) -> SignDataset:
    """Rebuild the active dataset to exactly `glosses` (the templated set), pulling
    difficulty from the loaded catalog (signs.json) and defaulting for any gloss the
    catalog doesn't list. Guarantees every streamed word is recognizable. The
    version is content-derived so a client can detect a mismatch."""
    global _dataset
    catalog = {e.word: e.difficulty for e in get_dataset().entries}
    ordered = sorted(glosses)
    if not ordered:
        raise ValueError("cannot restrict dataset to an empty gloss set")
    entries = tuple(
        SignEntry(sign_id=g, word=g, difficulty=catalog.get(g, DEFAULT_DIFFICULTY))
        for g in ordered
    )
    digest = hashlib.sha1(
        ";".join(f"{e.word}:{e.difficulty}" for e in entries).encode()
    ).hexdigest()[:8]
    _dataset = SignDataset(version=f"asl-{len(entries)}-{digest}", entries=entries)
    return _dataset


# --- deterministic word stream ----------------------------------------------
#
# Both client and server must generate the IDENTICAL word at a given index from a
# shared seed. We pin an explicit SplitMix64 mixer (not a language built-in, which
# differ across runtimes) so the JS client can reproduce it exactly. word_at is
# stateless per index, so a player can be scored at any position independently.


def _splitmix64(seed: int, index: int) -> int:
    z = (seed + index * 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def word_at(seed: int, index: int, dataset: SignDataset | None = None) -> SignEntry:
    ds = dataset or get_dataset()
    return ds.entries[_splitmix64(seed, index) % len(ds.entries)]


def word_iter(seed: int):
    """Infinite generator of words from the seeded stream — the prompt source for
    a player's recognition session (asl.Session consumes an Iterator[str])."""
    i = 0
    while True:
        yield word_at(seed, i).word
        i += 1


def random_seed() -> int:
    """A fresh, unpredictable seed for a match's word stream (31-bit, JS-safe)."""
    return secrets.randbits(31)
