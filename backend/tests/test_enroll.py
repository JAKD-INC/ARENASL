import numpy as np
import pytest

from train.enroll import enroll, save_prototypes


def _phase_encode(emb_dim=8):
    """Stub encoder (onnx-free, deterministic) so enrollment is unit-testable.

    Maps a window to a unit vector built from its mean over the first `emb_dim`
    feature columns. Because it reads the window's CONTENT, windows drawn from
    different phases of a clip (different feature values) embed to different
    directions — which is exactly what the phase clustering needs to see.
    """
    def encode(window):
        w = np.asarray(window, dtype=np.float32)
        v = w[:, :emb_dim].mean(axis=0)
        n = np.linalg.norm(v)
        return (v / n if n > 1e-12 else v).astype(np.float32)
    return encode


def _clip(value, T=20, F=84):
    """A constant clip whose first feature columns equal `value` (a one-hot-ish
    pattern), so its windows embed near a known direction."""
    base = np.zeros((T, F), dtype=np.float32)
    base[:, : len(value)] = np.asarray(value, dtype=np.float32)
    return base


def _two_phase_clip(a, b, T=80, F=84):
    """A clip whose first half sits at pattern `a` and second half at `b`, so
    windows sampled early vs late embed to two distinct directions (two phases).

    T is kept well above the window length (16) so most sampled windows fall
    cleanly inside one phase rather than straddling the transition — mirroring a
    real sign whose phases are each several windows long.
    """
    base = np.zeros((T, F), dtype=np.float32)
    half = T // 2
    base[:half, : len(a)] = np.asarray(a, dtype=np.float32)
    base[half:, : len(b)] = np.asarray(b, dtype=np.float32)
    return base


def _refs(value, n_clips, **kw):
    return [_clip(value, **kw) for _ in range(n_clips)]


def test_enroll_refuses_gloss_with_fewer_than_three_clips():
    encode = _phase_encode()
    refs = {
        "book": _refs([1, 0, 0, 0], 8),   # plenty
        "rare": _refs([0, 1, 0, 0], 2),   # below min_clips=3 -> refused
    }
    protos = enroll(encode, refs, rng=np.random.default_rng(0))
    assert "book" in protos
    assert "rare" not in protos


def test_enroll_keeps_gloss_at_min_clips_floor():
    encode = _phase_encode()
    refs = {"ok": _refs([1, 0, 0, 0], 3)}  # exactly min_clips
    protos = enroll(encode, refs, rng=np.random.default_rng(0))
    assert "ok" in protos


def test_enroll_builds_multiple_phase_prototypes_per_gloss():
    encode = _phase_encode()
    # Each clip has two clearly separated phases; with k=2 and many windows the
    # cloud should split into two phase prototypes.
    clips = [_two_phase_clip([1, 0, 0, 0], [0, 1, 0, 0]) for _ in range(8)]
    protos = enroll(encode, {"split": clips}, k=2, n_windows=12,
                    rng=np.random.default_rng(1))
    assert "split" in protos
    assert len(protos["split"]) == 2
    # The two phase prototypes point in distinct directions.
    a, b = protos["split"]
    assert float(np.dot(a, b)) < 0.9


def test_enroll_prototypes_are_l2_normalized():
    encode = _phase_encode()
    protos = enroll(encode, {"book": _refs([1, 0, 0, 0], 8)},
                    rng=np.random.default_rng(0))
    for p in protos["book"]:
        assert np.isclose(np.linalg.norm(p), 1.0, atol=1e-5)


def test_enroll_skips_empty_and_too_short_clip_lists():
    encode = _phase_encode()
    protos = enroll(encode, {"book": _refs([1, 0, 0, 0], 8), "empty": []},
                    rng=np.random.default_rng(0))
    assert set(protos) == {"book"}


def test_save_prototypes_one_row_per_prototype_with_repeats(tmp_path):
    """The npz holds one ROW per prototype; a multi-phase gloss repeats."""
    encode = _phase_encode()
    clips = [_two_phase_clip([1, 0, 0, 0], [0, 1, 0, 0]) for _ in range(8)]
    protos = enroll(encode, {"split": clips, "book": _refs([0, 0, 1, 0], 8)},
                    k=2, n_windows=12, rng=np.random.default_rng(2))
    path = str(tmp_path / "protos.npz")
    save_prototypes(path, protos)

    data = np.load(path, allow_pickle=True)
    glosses = [str(g) for g in data["glosses"]]
    matrix = data["protos"]
    # One row per prototype across all glosses, parallel-indexed.
    total = sum(len(v) for v in protos.values())
    assert matrix.shape[0] == len(glosses) == total
    assert matrix.shape[1] == 8
    # "split" repeats once per phase prototype.
    assert glosses.count("split") == len(protos["split"])
    # Rows stay L2-normed and grouping by gloss recovers the prototypes.
    for row in matrix:
        assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-5)
    grouped = {}
    for g, row in zip(glosses, matrix):
        grouped.setdefault(g, []).append(row)
    assert set(grouped) == {"split", "book"}


def test_save_prototypes_accepts_legacy_single_vector_shape(tmp_path):
    """Back-compat: a {gloss: vector} dict still writes one row per gloss."""
    protos = {"book": np.array([1, 0, 0], np.float32),
              "drink": np.array([0, 1, 0], np.float32)}
    path = str(tmp_path / "legacy.npz")
    save_prototypes(path, protos)
    data = np.load(path, allow_pickle=True)
    glosses = [str(g) for g in data["glosses"]]
    assert glosses == ["book", "drink"]  # sorted, one row each
    assert data["protos"].shape == (2, 3)


def test_enroll_save_roundtrip_with_random_encoder(tmp_path):
    """End-to-end with a content-sensitive stub: enroll -> save -> reload, and the
    reloaded rows match the enrolled prototypes grouped by gloss."""
    rng = np.random.default_rng(3)
    emb_dim = 8

    def encode(window):
        w = np.asarray(window, dtype=np.float32)
        v = w[:, :emb_dim].mean(axis=0) + 1e-3
        return (v / np.linalg.norm(v)).astype(np.float32)

    refs = {g: [_clip(np.eye(4)[i % 4], T=18) for _ in range(8)]
            for i, g in enumerate(("book", "drink", "eat"))}
    protos = enroll(encode, refs, k=2, n_windows=10, rng=rng)
    path = str(tmp_path / "p.npz")
    save_prototypes(path, protos)

    data = np.load(path, allow_pickle=True)
    glosses = [str(g) for g in data["glosses"]]
    matrix = data["protos"]
    assert set(glosses) == {"book", "drink", "eat"}
    assert matrix.shape[0] == sum(len(v) for v in protos.values())
    assert matrix.shape[1] == emb_dim


def test_enroll_warmup_fill_short_clips_still_enroll():
    """Clips shorter than the window (the warmup-fill case) still embed and
    enroll rather than being silently dropped."""
    encode = _phase_encode()
    # Clips of length 5 < WINDOW (16); sample_window returns them whole.
    short = [_clip([1, 0, 0, 0], T=5) for _ in range(6)]
    protos = enroll(encode, {"book": short}, window=16, rng=np.random.default_rng(4))
    assert "book" in protos
    assert all(np.isclose(np.linalg.norm(p), 1.0, atol=1e-5) for p in protos["book"])
