import numpy as np
import torch

from train.model import MotionEncoder


def test_forward_shapes_batch():
    m = MotionEncoder(in_dim=84, emb_dim=128, num_classes=5)
    x = torch.randn(4, 16, 84)
    emb, logits = m(x)
    assert emb.shape == (4, 128)
    assert logits.shape == (4, 5)


def test_embedding_is_l2_normalized():
    m = MotionEncoder(emb_dim=64, num_classes=3)
    x = torch.randn(3, 10, 84)
    emb = m.embed(x)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_handles_batch_one():
    m = MotionEncoder(num_classes=4)
    x = torch.randn(1, 7, 84)
    emb, logits = m(x)
    assert emb.shape == (1, 128)
    assert logits.shape == (1, 4)


def test_handles_variable_T():
    m = MotionEncoder(num_classes=4)
    for T in (1, 5, 33, 100):
        emb, logits = m(torch.randn(1, T, 84))
        assert emb.shape == (1, 128)
        assert torch.isfinite(emb).all()


def test_embed_rejects_wrong_rank():
    m = MotionEncoder(num_classes=2)
    try:
        m.embed(torch.randn(8, 84))
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-3D input")


def test_order_aware_time_reverse_distinct():
    """A sequence and its time-reverse must embed distinctly (cosine < 0.9).

    Mean-pooling the bi-GRU is order-blind; order-aware pooling (concat of final
    fwd+bwd hidden states) must distinguish a sign from its reverse. Use a clearly
    directional (monotonic) sequence so a correct encoder separates the two.
    """
    torch.manual_seed(0)
    m = MotionEncoder(emb_dim=128, num_classes=3)
    m.eval()
    T = 16
    ramp = torch.linspace(-1.0, 1.0, T).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    seq = ramp.expand(1, T, 84).contiguous()
    rev = torch.flip(seq, dims=[1])
    with torch.no_grad():
        emb = m.embed(seq)
        emb_rev = m.embed(rev)
    cos = torch.nn.functional.cosine_similarity(emb, emb_rev, dim=1).item()
    assert cos < 0.9, f"time-reverse embedding too similar (cos={cos:.4f}); pooling is order-blind"


def test_onnx_exportable_dynamic_T(tmp_path):
    import onnx
    import onnxruntime as ort
    from train.train import export_onnx

    m = MotionEncoder(emb_dim=32, num_classes=3)
    path = str(tmp_path / "enc.onnx")
    export_onnx(m, path)
    onnx.checker.check_model(onnx.load(path))

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    # Run two different T values through the same graph (dynamic axis).
    for T in (4, 20):
        out = sess.run(None, {name: np.random.randn(1, T, 84).astype(np.float32)})[0]
        assert out.shape == (1, 32)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
