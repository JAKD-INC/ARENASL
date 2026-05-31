import os

import numpy as np

from train.dataset import WINDOW_SIZE, SequenceDataset
from train.model import MotionEncoder
from train.train import collate, evaluate, export_onnx, train


class _SynthDS:
    """Tiny variable-T synthetic dataset: each class has a distinct constant
    offset so a few epochs can fit it. Yields (window (T,84) float32, label)."""

    def __init__(self, num_classes=4, per_class=6, seed=0):
        rng = np.random.default_rng(seed)
        self.items = []
        for c in range(num_classes):
            for _ in range(per_class):
                T = int(rng.integers(5, 15))
                base = np.full((T, 84), float(c), dtype=np.float32)
                base += rng.standard_normal((T, 84)).astype(np.float32) * 0.01
                self.items.append((base, c))
        self.num_classes = num_classes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_collate_pads_variable_T():
    batch = [
        (np.ones((3, 84), dtype=np.float32), 0),
        (np.ones((7, 84), dtype=np.float32), 1),
    ]
    x, y = collate(batch)
    assert tuple(x.shape) == (2, 7, 84)
    assert tuple(y.shape) == (2,)


def test_train_runs_on_synthetic_and_returns_metrics():
    ds = _SynthDS(num_classes=4, per_class=6)
    model = MotionEncoder(emb_dim=32, num_classes=4)
    model, metrics = train(model, ds, ds, epochs=2, batch_size=4)
    assert set(metrics) == {"top1", "top5"}
    assert 0.0 <= metrics["top1"] <= 1.0
    assert 0.0 <= metrics["top5"] <= 1.0


def test_train_can_fit_easy_synthetic():
    ds = _SynthDS(num_classes=3, per_class=8, seed=1)
    model = MotionEncoder(emb_dim=32, num_classes=3)
    model, metrics = train(model, ds, ds, epochs=15, batch_size=6, lr=5e-3)
    # On a trivially separable synthetic set the model should beat chance (1/3).
    assert metrics["top1"] > 0.34


def test_evaluate_empty_loader():
    from torch.utils.data import DataLoader
    model = MotionEncoder(num_classes=3)
    loader = DataLoader(_SynthDS(num_classes=3, per_class=0), batch_size=4, collate_fn=collate)
    metrics = evaluate(model, loader)
    assert metrics == {"top1": 0.0, "top5": 0.0}


def test_train_one_epoch_then_export_and_reload_onnx(tmp_path):
    """Train a MotionEncoder for one epoch on a tiny synthetic set, export the
    encoder to ONNX, then reload with onnxruntime and confirm the embedding output
    is (1, 128) for a (1, T, 84) input with a dynamic T axis (the live contract)."""
    import onnxruntime as ort

    ds = _SynthDS(num_classes=3, per_class=4, seed=2)
    model = MotionEncoder(emb_dim=128, num_classes=3)
    model, metrics = train(model, ds, ds, epochs=1, batch_size=4)
    assert set(metrics) == {"top1", "top5"}

    path = os.path.join(tmp_path, "encoder.onnx")
    export_onnx(model, path)
    assert os.path.exists(path)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # A different T than the export example (8) exercises the dynamic time axis.
    x = np.random.randn(1, 11, 84).astype(np.float32)
    emb = sess.run([out_name], {in_name: x})[0]
    assert tuple(emb.shape) == (1, 128)
    # The exported graph keeps the L2-normalization, so the embedding is unit-norm.
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)


def _windowed_samples(num_classes=3, per_class=6, T=64, seed=3):
    """Synthetic cache-shaped samples whose raw sequences are LONGER than the live
    window length L, so the training path must window them down to <= L frames."""
    rng = np.random.default_rng(seed)
    samples = []
    for c in range(num_classes):
        for j in range(per_class):
            seq = np.full((T, 84), float(c), dtype=np.float32)
            seq += rng.standard_normal((T, 84)).astype(np.float32) * 0.01
            samples.append({"gloss": f"g{c}", "signer_id": f"s{c}_{j}", "seq": seq})
    return samples


def test_train_windows_are_at_most_L():
    """The live matcher feeds the encoder a fixed-L sliding window, so the train
    side must train on windowed samples (via dataset.sample_window): every window
    yielded by the training dataset is at most L frames long, even though the raw
    clips are far longer. Some windows are shorter (the warmup-fill phase)."""
    L = WINDOW_SIZE
    samples = _windowed_samples(T=L * 4)
    ds = SequenceDataset(samples, train=True, window=L)
    lengths = [ds[i][0].shape[0] for i in range(len(ds)) for _ in range(8)]
    assert lengths, "expected windows to inspect"
    assert all(1 <= n <= L for n in lengths), (
        f"training windows must be in [1, {L}]; got {sorted(set(lengths))}"
    )
    # The warmup-fill path means at least some windows are strictly shorter than L,
    # and the fixed-L path means at least one window reaches the full length L.
    assert any(n < L for n in lengths)
    assert any(n == L for n in lengths)


def test_train_runs_on_windowed_dataset():
    """End-to-end: train() consumes the windowed SequenceDataset (clips longer than
    L, sampled down to <=L windows) and returns the metric dict."""
    L = WINDOW_SIZE
    samples = _windowed_samples(T=L * 3)
    train_ds = SequenceDataset(samples, train=True, window=L)
    val_ds = SequenceDataset(samples, train=False, window=L)
    model = MotionEncoder(emb_dim=32, num_classes=train_ds.num_classes)
    model, metrics = train(model, train_ds, val_ds, epochs=2, batch_size=4)
    assert set(metrics) == {"top1", "top5"}
    assert 0.0 <= metrics["top1"] <= 1.0
