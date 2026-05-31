"""Temporal motion encoder for ASL windows.

A small bi-GRU reads a (B, T, 84) sequence of hands-xy features, mean-pools over
the time axis, projects to a 128-d embedding (L2-normalized), and a linear head
produces class logits. The embedding is what live and reference windows are
compared by (cosine); the head is only used to train it.

Kept deliberately tiny (~1-2M params) and onnx.export-able with a dynamic T axis
so it runs at ~ms/window on CPU in the live loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    def __init__(self, in_dim: int = 84, emb_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        hidden = 128
        # Bidirectional GRU over the time axis. batch_first so x is (B, T, in_dim).
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        # Mean-pooled bi-GRU output (2*hidden) -> emb_dim, then L2-normalized.
        self.proj = nn.Linear(2 * hidden, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, in_dim) float32 -> (B, emb_dim) L2-normalized embedding.

        Handles B == 1 and variable T. Mean-pooling over T (not the last hidden
        state) makes the embedding robust to where in the window a sign falls.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, {self.in_dim}); got shape {tuple(x.shape)}")
        out, _ = self.gru(x)            # (B, T, 2*hidden)
        pooled = out.mean(dim=1)        # (B, 2*hidden)
        emb = self.proj(pooled)         # (B, emb_dim)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, x: torch.Tensor):
        """(B, T, in_dim) -> (embedding (B, emb_dim), logits (B, num_classes))."""
        emb = self.embed(x)
        logits = self.classifier(emb)
        return emb, logits
