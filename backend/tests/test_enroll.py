import numpy as np

from train.enroll import enroll, save_prototypes


def _identity_encode(emb_dim=8):
    """Encode = mean over time of the first emb_dim feature columns, L2-normed.
    Deterministic and onnx-free so enrollment is unit-testable."""
    def encode(window):
        w = np.asarray(window, dtype=np.float32)
        v = w[:, :emb_dim].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v
    return encode


def test_enroll_returns_one_prototype_per_gloss():
    encode = _identity_encode()
    refs = {
        "book": [np.ones((5, 84), dtype=np.float32)],
        "drink": [np.full((4, 84), 2.0, dtype=np.float32)],
    }
    protos = enroll(encode, refs)
    assert set(protos) == {"book", "drink"}
    for v in protos.values():
        assert v.shape == (8,)


def test_prototypes_are_l2_normalized():
    encode = _identity_encode()
    refs = {"book": [np.ones((5, 84), dtype=np.float32),
                     np.full((6, 84), 3.0, dtype=np.float32)]}
    protos = enroll(encode, refs)
    assert np.isclose(np.linalg.norm(protos["book"]), 1.0, atol=1e-5)


def test_prototype_is_mean_of_clip_embeddings():
    encode = _identity_encode(emb_dim=4)
    a = np.tile(np.array([1, 0, 0, 0] + [0] * 80, dtype=np.float32), (3, 1))
    b = np.tile(np.array([0, 1, 0, 0] + [0] * 80, dtype=np.float32), (3, 1))
    protos = enroll(encode, {"x": [a, b]})
    # mean of (1,0,0,0) and (0,1,0,0) -> (0.5,0.5,0,0) -> normed
    expected = np.array([1, 1, 0, 0], dtype=np.float32) / np.sqrt(2)
    assert np.allclose(protos["x"], expected, atol=1e-5)


def test_save_prototypes_roundtrip(tmp_path):
    encode = _identity_encode()
    protos = enroll(encode, {"book": [np.ones((5, 84), dtype=np.float32)],
                             "drink": [np.full((4, 84), 2.0, dtype=np.float32)]})
    path = str(tmp_path / "protos.npz")
    save_prototypes(path, protos)
    data = np.load(path, allow_pickle=True)
    glosses = [str(g) for g in data["glosses"]]
    assert glosses == ["book", "drink"]  # sorted
    assert data["protos"].shape == (2, 8)
    for i, g in enumerate(glosses):
        assert np.allclose(data["protos"][i], protos[g])


def test_enroll_skips_empty_clip_lists():
    encode = _identity_encode()
    protos = enroll(encode, {"book": [np.ones((5, 84), dtype=np.float32)], "empty": []})
    assert set(protos) == {"book"}
