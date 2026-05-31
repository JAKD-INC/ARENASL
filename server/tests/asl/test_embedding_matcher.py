import numpy as np
import pytest

from asl.embedding_matcher import EmbeddingMatcher


def _fixed_encode(vec):
    """An encode that ignores the window and returns a fixed L2-normed embedding,
    so we control the cosine exactly."""
    v = np.asarray(vec, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return lambda window: v


def _protos():
    book = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    drink = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return {"book": book, "drink": drink}


def _l2(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_strength_one_when_embedding_equals_prototype():
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), _protos())
    assert m.strength(np.zeros((5, 84), np.float32), "book") == pytest.approx(1.0)


def test_strength_half_when_orthogonal():
    m = EmbeddingMatcher(_fixed_encode([0.0, 1.0, 0.0]), _protos())
    # cosine 0 -> (0+1)/2 = 0.5
    assert m.strength(np.zeros((5, 84), np.float32), "book") == pytest.approx(0.5)


def test_strength_zero_when_opposite():
    m = EmbeddingMatcher(_fixed_encode([-1.0, 0.0, 0.0]), _protos())
    assert m.strength(np.zeros((5, 84), np.float32), "book") == pytest.approx(0.0)


def test_strength_in_unit_interval_for_random_encoders():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = EmbeddingMatcher(_fixed_encode(rng.standard_normal(3)), _protos())
        s = m.strength(np.zeros((3, 84), np.float32), "book")
        assert 0.0 <= s <= 1.0


def test_zero_embedding_scores_worst_not_neutral():
    """A degenerate (zero-norm) embedding is a missing/failed encode, so it should
    score 0.0 (worst), not the neutral 0.5 a cosine of 0 would yield."""
    m = EmbeddingMatcher(lambda window: np.zeros(3, np.float32), _protos())
    assert m.strength(np.zeros((5, 84), np.float32), "book") == pytest.approx(0.0)
    assert m.best_distance(np.zeros((5, 84), np.float32), "book") == pytest.approx(2.0)


def test_best_distance_is_one_minus_cosine():
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), _protos())
    assert m.best_distance(np.zeros((5, 84), np.float32), "book") == pytest.approx(0.0)
    assert m.best_distance(np.zeros((5, 84), np.float32), "drink") == pytest.approx(1.0)


def test_rank_orders_by_cosine_distance():
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), _protos())
    ranked = m.rank(np.zeros((5, 84), np.float32), k=2)
    assert [r["gloss"] for r in ranked] == ["book", "drink"]
    assert ranked[0]["distance"] <= ranked[1]["distance"]


def test_unknown_target_raises_keyerror():
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), _protos())
    with pytest.raises(KeyError):
        m.strength(np.zeros((5, 84), np.float32), "MISSING")


def test_multiple_prototypes_strength_picks_best_phase():
    """A gloss carries several phase prototypes; strength is the MAX over them, so a
    window matching ANY phase scores high even if it is far from the others."""
    # Two orthogonal phase prototypes for "book"; the embedding equals phase 2.
    book = np.stack([_l2([1.0, 0.0, 0.0]), _l2([0.0, 0.0, 1.0])], axis=0)
    protos = {"book": book, "drink": _l2([0.0, 1.0, 0.0])}
    m = EmbeddingMatcher(_fixed_encode([0.0, 0.0, 1.0]), protos)
    # cos to phase1 = 0 -> 0.5, cos to phase2 = 1 -> 1.0; max wins.
    assert m.strength(np.zeros((5, 84), np.float32), "book") == pytest.approx(1.0)
    assert m.best_distance(np.zeros((5, 84), np.float32), "book") == pytest.approx(0.0)


def test_multiple_prototypes_best_distance_is_min_over_phases():
    """best_distance = MIN (1 - cos) over the gloss's prototypes (nearest phase)."""
    # phase1 cos = 0 (dist 1.0), phase2 cos = 0.6 (dist 0.4); min = 0.4.
    book = np.stack([_l2([0.0, 1.0, 0.0]), _l2([0.6, 0.8, 0.0])], axis=0)
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), {"book": book})
    assert m.best_distance(np.zeros((5, 84), np.float32), "book") == pytest.approx(0.4)
    # strength uses the same best phase: (0.6 + 1) / 2 = 0.8.
    assert m.strength(np.zeros((5, 84), np.float32), "book") == pytest.approx(0.8)


def test_rank_uses_best_per_gloss_across_phases():
    """rank emits one entry per gloss, scored by its nearest prototype, not per row."""
    # "book" has a far phase and a close phase; its best (close) phase should win
    # over "drink"'s single moderate prototype.
    book = np.stack([_l2([0.0, -1.0, 0.0]), _l2([1.0, 0.0, 0.0])], axis=0)
    drink = _l2([0.7, 0.7, 0.0])
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), {"book": book, "drink": drink})
    ranked = m.rank(np.zeros((5, 84), np.float32), k=2)
    assert [r["gloss"] for r in ranked] == ["book", "drink"]
    assert {r["gloss"] for r in ranked} == {"book", "drink"}  # no duplicate gloss rows


def test_rank_encodes_window_once_not_per_gloss():
    """rank() must encode the window ONCE (then a single matmul over all prototype
    rows), not re-encode per gloss — otherwise it is unusable on the confirm path
    (1308 ONNX forward passes per call)."""
    calls = {"n": 0}

    def counting_encode(window):
        calls["n"] += 1
        return _l2([1.0, 0.0, 0.0])

    protos = {"book": _l2([1.0, 0.0, 0.0]), "drink": _l2([0.0, 1.0, 0.0]),
              "eat": _l2([0.0, 0.0, 1.0])}
    m = EmbeddingMatcher(counting_encode, protos)
    ranked = m.rank(np.zeros((5, 84), np.float32), k=3)
    assert calls["n"] == 1
    assert ranked[0]["gloss"] == "book"  # highest cosine still ranked first


def test_one_dim_window_raises_valueerror():
    """A flat (84,) frame is not a motion window; guard it instead of scoring noise."""
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), _protos())
    with pytest.raises(ValueError):
        m.strength(np.zeros(84, np.float32), "book")
    with pytest.raises(ValueError):
        m.best_distance(np.zeros(84, np.float32), "book")
    with pytest.raises(ValueError):
        m.rank(np.zeros(84, np.float32))


def test_three_dim_window_raises_valueerror():
    """A 3-D (B, T, 84) batch is not a single motion window: it would slip past
    from_files' batch-wrapping (which only adds a batch axis when ndim == 2) and feed
    an extra axis to the encoder. The guard must reject it, not just 1-D inputs."""
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), _protos())
    with pytest.raises(ValueError):
        m.strength(np.zeros((1, 5, 84), np.float32), "book")
    with pytest.raises(ValueError):
        m.best_distance(np.zeros((1, 5, 84), np.float32), "book")
    with pytest.raises(ValueError):
        m.rank(np.zeros((1, 5, 84), np.float32))


def test_no_motion_window_scores_worst_when_gated():
    """A MOTION embedding of a NON-MOVING window is meaningless: a no-hands frame
    (MediaPipe zero-fills an undetected hand) or hands held still produce a CONSTANT
    window that the encoder maps to ~1.0 cosine against most prototypes -> false
    confirms. With a motion floor, such a window must score worst (0.0) regardless
    of the encoder's cosine."""
    # encode would give cosine 1.0 (embedding == "book" prototype).
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), _protos(), min_motion=0.01)
    const = np.full((6, 84), 0.3, np.float32)  # zero temporal variation
    assert m.strength(const, "book") == pytest.approx(0.0)
    assert m.best_distance(const, "book") == pytest.approx(2.0)
    # A window WITH motion is scored normally — the gate must not fire on real signs.
    moving = const.copy()
    moving[::2] += 0.5  # temporal std >> 0.01
    assert m.strength(moving, "book") == pytest.approx(1.0)


def test_motion_gate_off_by_default_preserves_pure_cosine():
    """Default (no motion floor) keeps the pure-cosine contract: a constant window
    still scores by cosine, so direct construction is unchanged. The floor is a
    production guard switched on by from_files, not the default."""
    m = EmbeddingMatcher(_fixed_encode([1.0, 0.0, 0.0]), _protos())
    assert m.strength(np.zeros((5, 84), np.float32), "book") == pytest.approx(1.0)


def test_from_files_with_real_onnx(tmp_path):
    """End-to-end: export a real encoder, enroll synthetic clips, then match via
    onnxruntime through from_files."""
    from train.model import MotionEncoder
    from train.train import export_onnx
    from train.enroll import enroll, save_prototypes

    model = MotionEncoder(emb_dim=16, num_classes=3)
    onnx_path = str(tmp_path / "enc.onnx")
    export_onnx(model, onnx_path)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def encode(window):
        w = np.asarray(window, dtype=np.float32)[None, ...]
        return sess.run(None, {name: w})[0].reshape(-1)

    rng = np.random.default_rng(0)
    # Phase 1 DATA FLOOR: enroll() REFUSES a gloss with < 3 usable clips and logs
    # any below the data floor (8) as under-supported. Provide >= 8 clips/gloss so
    # every gloss is enrolled cleanly and its prototypes survive into the store.
    refs = {g: [rng.standard_normal((8, 84)).astype(np.float32) for _ in range(8)]
            for g in ("book", "drink", "eat")}
    protos = enroll(encode, refs)
    # Guard: enroll must have kept all three glosses (the matcher below indexes them).
    assert set(protos) == {"book", "drink", "eat"}
    protos_path = str(tmp_path / "protos.npz")
    save_prototypes(protos_path, protos)

    m = EmbeddingMatcher.from_files(onnx_path, protos_path)
    win = refs["book"][0]
    s = m.strength(win, "book")
    assert 0.0 <= s <= 1.0
    ranked = m.rank(win, k=3)
    assert len(ranked) == 3
