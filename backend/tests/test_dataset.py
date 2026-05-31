import numpy as np

from train.dataset import SequenceDataset, signer_split


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
