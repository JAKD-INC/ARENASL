"""Train the MotionEncoder (cross-entropy on the classifier head) and export the
encoder to ONNX with a dynamic time axis.

The embedding is the byproduct we actually ship: a good classifier forces the
penultimate L2-normalized embedding to cluster by sign, which is what the live
matcher compares by cosine. `train` runs on tiny synthetic tensors (unit-tested);
`main` is the offline CLI driving it on a built cache.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.model import MotionEncoder


def collate(batch):
    """Pad variable-T windows in a batch to a common length -> (B, Tmax, 84)."""
    windows, labels = zip(*batch)
    Tmax = max(w.shape[0] for w in windows)
    B = len(windows)
    D = windows[0].shape[1]
    x = np.zeros((B, Tmax, D), dtype=np.float32)
    for i, w in enumerate(windows):
        x[i, : w.shape[0]] = w
    return (
        torch.from_numpy(x),
        torch.tensor(labels, dtype=torch.long),
    )


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices  # (B, k)
    correct = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct


def evaluate(model: MotionEncoder, loader) -> dict:
    model.eval()
    n, top1_sum, top5_sum = 0, 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            _, logits = model(x)
            b = y.shape[0]
            top1_sum += _topk_accuracy(logits, y, 1) * b
            top5_sum += _topk_accuracy(logits, y, 5) * b
            n += b
    if n == 0:
        return {"top1": 0.0, "top5": 0.0}
    return {"top1": top1_sum / n, "top5": top5_sum / n}


def train(model, train_ds, val_ds, epochs: int = 5, lr: float = 1e-3,
          batch_size: int = 16, device: str = "cpu"):
    """Cross-entropy training over the classifier logits. Returns (model, metrics)
    where metrics = {"top1", "top5"} on the validation set."""
    model.to(device)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _, logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

    metrics = evaluate(model, val_loader)
    return model, metrics


def export_onnx(model: MotionEncoder, path: str):
    """Export the encoder (embed path) to ONNX with a dynamic T axis.

    The exported graph maps a (1, T, 84) input -> embedding (1, emb_dim). Only the
    embedding is needed live, so we export a thin module wrapping `embed`."""
    model.eval()

    class _EmbedOnly(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, x):
            return self.enc.embed(x)

    wrapper = _EmbedOnly(model)
    example = torch.zeros(1, 8, model.in_dim, dtype=torch.float32)
    # Force the stable TorchScript exporter (dynamo=False). The installed torch
    # (2.12) defaults to the dynamo path, which fails to solve the dynamic-T shape
    # constraints for our bi-GRU; the legacy exporter handles GRU + variable T +
    # batch_size=1 cleanly. Inference is batch_size=1 (one window per frame in the
    # live matcher), which is what `embedding`'s dynamic B axis plus the GRU export
    # path supports here. (On older torch <2.5, dynamo=False is also the default,
    # so passing it explicitly is a safe no-op there.)
    torch.onnx.export(
        wrapper,
        example,
        path,
        input_names=["window"],
        output_names=["embedding"],
        dynamic_axes={"window": {1: "T"}, "embedding": {0: "B"}},
        opset_version=17,
        dynamo=False,
    )
    return path


def _load_cache(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    seqs = data["seqs"]
    glosses = data["glosses"]
    signers = data["signer_ids"]
    return [
        {"gloss": str(g), "signer_id": str(s), "seq": np.asarray(seq, dtype=np.float32)}
        for seq, g, s in zip(seqs, glosses, signers)
    ]


def main():
    from train.dataset import SequenceDataset, signer_split

    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help=".npz from dataset.build_cache")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--onnx", default="data/encoder.onnx")
    a = p.parse_args()

    samples = _load_cache(a.cache)
    train_s, val_s = signer_split(samples, a.val_frac)
    train_ds = SequenceDataset(train_s, train=True, window=a.window)
    val_ds = SequenceDataset(val_s, train=False, window=a.window)
    model = MotionEncoder(emb_dim=a.emb_dim, num_classes=train_ds.num_classes)
    model, metrics = train(model, train_ds, val_ds, epochs=a.epochs)
    print(f"val top1={metrics['top1']:.3f} top5={metrics['top5']:.3f}")
    export_onnx(model, a.onnx)
    print(f"exported encoder -> {a.onnx}")


if __name__ == "__main__":
    main()
