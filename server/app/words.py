"""Sign dataset loader (the `word_strength` source) and word-stream generator.

The dataset maps every sign to a word and a `difficulty` score — that difficulty
*is* the word strength. It's static reference data, loaded into memory once at
startup (fail fast if missing/unparseable) and shared with clients via GET /signs
so client and server use the identical ordered set.

Phase 1a implements loading + lookup. The seeded PRNG word stream lands in 1e.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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
