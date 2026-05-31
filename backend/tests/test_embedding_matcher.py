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
    refs = {g: [rng.standard_normal((8, 84)).astype(np.float32) for _ in range(2)]
            for g in ("book", "drink", "eat")}
    protos = enroll(encode, refs)
    protos_path = str(tmp_path / "protos.npz")
    save_prototypes(protos_path, protos)

    m = EmbeddingMatcher.from_files(onnx_path, protos_path)
    win = refs["book"][0]
    s = m.strength(win, "book")
    assert 0.0 <= s <= 1.0
    ranked = m.rank(win, k=3)
    assert len(ranked) == 3
