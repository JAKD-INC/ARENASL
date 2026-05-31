import numpy as np

from train.dataset import (
    WINDOW_SIZE,
    SequenceDataset,
    sample_window,
    signer_split,
)


def _sample(gloss, signer, T=20):
    rng = np.random.default_rng(hash((gloss, signer, T)) % (2**32))
    return {"gloss": gloss, "signer_id": signer,
            "seq": rng.standard_normal((T, 84)).astype(np.float32)}


def _samples():
    out = []
    for g in ("book", "drink", "eat"):
        for sg in ("s1", "s2", "s3", "s4", "s5"):
            out.append(_sample(g, sg))
    return out


def test_signer_split_no_signer_on_both_sides():
    train, val = signer_split(_samples(), val_frac=0.2)
    train_signers = {s["signer_id"] for s in train}
    val_signers = {s["signer_id"] for s in val}
    assert train_signers.isdisjoint(val_signers)
    assert len(train) > 0 and len(val) > 0


def test_signer_split_covers_all_samples():
    samples = _samples()
    train, val = signer_split(samples, val_frac=0.3)
    assert len(train) + len(val) == len(samples)


def test_signer_split_many_samples_one_signer_keeps_val_nonempty():
    # All samples share signer s1: both invariants cannot hold, so the split
    # must still leave both sides non-empty rather than returning an empty val.
    samples = [_sample(g, "s1") for g in ("book", "drink", "eat", "go", "help")]
    train, val = signer_split(samples, val_frac=0.2)
    assert len(train) > 0 and len(val) > 0
    assert len(train) + len(val) == len(samples)


def test_signer_split_multi_signer_always_both_nonempty():
    # Tiny pool (2 signers) at a low val_frac must still hold one signer out.
    samples = [_sample("book", "s1"), _sample("book", "s2")]
    train, val = signer_split(samples, val_frac=0.01)
    assert len(train) > 0 and len(val) > 0
    train_signers = {s["signer_id"] for s in train}
    val_signers = {s["signer_id"] for s in val}
    assert train_signers.isdisjoint(val_signers)


def test_signer_split_empty_input():
    assert signer_split([], val_frac=0.2) == ([], [])


def test_signer_split_single_sample_single_signer_keeps_train_nonempty():
    # One sample, one signer: both sides cannot be non-empty (a lone sample
    # can't be in two places), so the sample must stay on the train side rather
    # than handing back an empty train set.
    samples = [_sample("book", "s1")]
    train, val = signer_split(samples, val_frac=0.2)
    assert len(train) == 1 and len(val) == 0
    assert train[0]["signer_id"] == "s1"


def test_dataset_yields_window_and_int_label():
    samples = _samples()
    ds = SequenceDataset(samples, train=False, window=64)
    window, label = ds[0]
    assert window.dtype == np.float32
    assert window.ndim == 2 and window.shape[1] == 84
    assert isinstance(label, int)
    assert 0 <= label < ds.num_classes


def test_dataset_caps_window_length():
    samples = [_sample("book", "s1", T=200)]
    ds = SequenceDataset(samples, train=False, window=32)
    window, _ = ds[0]
    assert window.shape[0] <= 32


def test_labels_consistent_with_sorted_glosses():
    ds = SequenceDataset(_samples(), train=False)
    assert ds.glosses == sorted(ds.glosses)
    assert ds.num_classes == 3
    # every gloss maps to a stable index
    for i, g in enumerate(ds.glosses):
        assert ds.label_of[g] == i


def test_train_side_augments_within_window():
    samples = [_sample("book", "s1", T=50)]
    ds = SequenceDataset(samples, train=True, window=32, seed=3)
    for i in range(10):
        window, label = ds[0]
        assert 1 <= window.shape[0] <= 32
        assert window.shape[1] == 84
        assert np.isfinite(window).all()
        assert label == 0


def test_len_matches_samples():
    samples = _samples()
    ds = SequenceDataset(samples, train=False)
    assert len(ds) == len(samples)


# --- sample_window (fixed-L windowing, matching the live sliding window) ---

def test_sample_window_returns_exactly_L_when_seq_longer():
    rng = np.random.default_rng(0)
    seq = np.arange(40 * 84, dtype=np.float32).reshape(40, 84)
    w = sample_window(seq, 16, rng)
    assert w.shape == (16, 84)
    assert w.dtype == np.float32


def test_sample_window_is_a_contiguous_subwindow():
    rng = np.random.default_rng(1)
    seq = np.arange(40 * 84, dtype=np.float32).reshape(40, 84)
    w = sample_window(seq, 16, rng)
    # The window must be a contiguous slice of the source (rows step by 84).
    start = int(w[0, 0]) // 84
    assert np.array_equal(w, seq[start:start + 16])


def test_sample_window_random_start_varies():
    rng = np.random.default_rng(2)
    seq = np.arange(64 * 84, dtype=np.float32).reshape(64, 84)
    starts = {int(sample_window(seq, 16, rng)[0, 0]) // 84 for _ in range(30)}
    assert len(starts) > 1  # random start actually moves


def test_sample_window_allows_shorter_than_L():
    # Warmup-fill: T < L must return the whole (shorter) sequence, never pad.
    rng = np.random.default_rng(3)
    seq = np.arange(5 * 84, dtype=np.float32).reshape(5, 84)
    w = sample_window(seq, 16, rng)
    assert w.shape == (5, 84)
    assert np.array_equal(w, seq)


def test_sample_window_never_empty():
    rng = np.random.default_rng(4)
    seq = np.zeros((1, 84), dtype=np.float32)
    assert sample_window(seq, 16, rng).shape[0] == 1


# --- SequenceDataset now samples the live window length ---

def test_dataset_default_window_is_live_window_size():
    ds = SequenceDataset(_samples(), train=False)
    assert ds.window == WINDOW_SIZE


def test_val_window_capped_at_L():
    samples = [_sample("book", "s1", T=200)]
    ds = SequenceDataset(samples, train=False, window=16)
    window, _ = ds[0]
    assert window.shape[0] <= 16


def test_train_side_emits_some_short_warmup_windows():
    # With SHORT_WINDOW_PROB > 0 the train side must sometimes emit T<L windows
    # (mimicking the live buffer before it fills), not always exactly L.
    samples = [_sample("book", "s1", T=200)]
    ds = SequenceDataset(samples, train=True, window=16, seed=7)
    lengths = {ds[0][0].shape[0] for _ in range(60)}
    assert max(lengths) <= 16
    assert any(L < 16 for L in lengths)


# --- signer_split asserts disjointness via the smallest-group fallback path ---

def test_signer_split_disjoint_via_fallback_holds():
    # val_frac=0 would normally send nothing to val; the smallest-group fallback
    # then holds one whole signer out, and the contract assertion must still pass
    # (no signer on both sides).
    train, val = signer_split(_samples(), val_frac=0.0)
    assert len(train) > 0 and len(val) > 0
    assert {s["signer_id"] for s in train}.isdisjoint({s["signer_id"] for s in val})


# --- build_cache signer join (no network / MediaPipe: stub the heavy imports) ---

def _stub_extraction(monkeypatch, frames=3):
    """Install lightweight stand-ins for build.extract / asl.features so
    build_cache's in-function imports never pull in cv2 / MediaPipe.

    build_cache does `from build.extract import extract_frames` and
    `from asl.features import normalize_frame` at call time, so a stub module in
    sys.modules satisfies both without touching the network or a webcam stack.
    """
    import sys
    import types

    extract_mod = types.ModuleType("build.extract")
    extract_mod.extract_frames = lambda path: [
        np.zeros(147, dtype=np.float32) for _ in range(frames)
    ]
    features_mod = types.ModuleType("asl.features")
    features_mod.normalize_frame = lambda f: np.asarray(f, dtype=np.float32).reshape(49, 3)
    monkeypatch.setitem(sys.modules, "build.extract", extract_mod)
    monkeypatch.setitem(sys.modules, "asl.features", features_mod)


def test_build_cache_joins_signer_by_video(tmp_path, monkeypatch):
    import train.dataset as ds_mod

    _stub_extraction(monkeypatch)
    clip_paths = {"book": [("vidA", "/fake/a.mp4"), ("vidB", "/fake/b.mp4")]}
    signer_by_video = {"vidA": 7}  # vidB intentionally unmatched
    out = tmp_path / "cache.npz"
    samples = ds_mod.build_cache(
        ["book"], clip_paths, str(out), signer_by_video=signer_by_video
    )
    signers = sorted(str(s["signer_id"]) for s in samples)
    # vidA joined to signer 7; vidB had no match -> its own unique signer bucket.
    assert "7" in signers
    assert any(sid.startswith("vid:vidB") for sid in signers)
    assert out.exists()


def test_build_cache_written_npz_roundtrips_signers(tmp_path, monkeypatch):
    import train.dataset as ds_mod

    _stub_extraction(monkeypatch)
    clip_paths = {"eat": [("v1", "/fake/1.mp4")]}
    out = tmp_path / "c.npz"
    ds_mod.build_cache(["eat"], clip_paths, str(out), signer_by_video={"v1": "S9"})
    data = np.load(out, allow_pickle=True)
    assert list(data["glosses"]) == ["eat"]
    assert list(data["signer_ids"]) == ["S9"]


def test_build_cache_features_are_84_wide(tmp_path, monkeypatch):
    # The stored sequence must be reduced to the 84-d hand-xy match features so
    # train/enroll/live all share one feature space.
    import train.dataset as ds_mod

    _stub_extraction(monkeypatch)
    out = tmp_path / "c.npz"
    samples = ds_mod.build_cache(
        ["eat"], {"eat": [("v1", "/fake/1.mp4")]}, str(out),
        signer_by_video={"v1": "S1"},
    )
    assert samples[0]["seq"].shape[1] == 84
