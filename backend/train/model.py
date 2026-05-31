"""Temporal motion encoder for ASL windows.

A small bi-GRU reads a (B, T, 84) sequence of hands-xy features, takes the
ORDER-AWARE pooled state (the concatenated final forward+backward hidden states
of the top GRU layer), projects to a 128-d embedding (L2-normalized), and a
linear head produces class logits. The embedding is what live and reference
windows are compared by (cosine); the head is only used to train it.

Order-aware pooling (vs mean over time) matters because signs ARE motion: a sign
and its time-reverse must not embed alike. The final forward hidden summarizes the
sequence read left-to-right and the final backward hidden right-to-left, so their
concatenation encodes temporal direction.

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
        self.hidden = hidden
        self.num_layers = 2
        # Bidirectional GRU over the time axis. batch_first so x is (B, T, in_dim).
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        # Order-aware pooled state (concat of final fwd+bwd hidden, 2*hidden)
        # -> emb_dim, then L2-normalized.
        self.proj = nn.Linear(2 * hidden, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, in_dim) float32 -> (B, emb_dim) L2-normalized embedding.

        Handles B == 1 and variable T. Pooling is ORDER-AWARE: it concatenates the
        final forward and final backward hidden states of the top GRU layer rather
        than averaging the per-step outputs over time. This keeps the embedding
        sensitive to temporal direction (a sign vs its reverse), which mean-pooling
        is blind to.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, {self.in_dim}); got shape {tuple(x.shape)}")
        # h_n: (num_layers * num_directions, B, hidden). Layers are stacked then
        # directions interleaved: with num_layers=2 the rows are
        # [L0-fwd, L0-bwd, L1-fwd, L1-bwd], so the TOP layer's forward/backward hidden
        # are always the last two slices regardless of num_layers (-2 = top fwd,
        # -1 = top bwd). Keep this indexing if num_layers changes.
        _, h_n = self.gru(x)
        h_fwd = h_n[-2]                 # (B, hidden) top-layer forward final state
        h_bwd = h_n[-1]                 # (B, hidden) top-layer backward final state
        pooled = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2*hidden), order-aware
        emb = self.proj(pooled)         # (B, emb_dim)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, x: torch.Tensor):
        """(B, T, in_dim) -> (embedding (B, emb_dim), logits (B, num_classes))."""
        emb = self.embed(x)
        logits = self.classifier(emb)
        return emb, logits
