import numpy as np

from cascade.federated.secure_agg import SecureAggregator, pairwise_masks


def test_masks_sum_to_zero():
    masks = pairwise_masks(7, 16, seed=3)
    total = np.sum(np.vstack(masks), axis=0)
    assert np.allclose(total, 0.0, atol=1e-9)


def test_aggregate_equals_sum_of_raw():
    rng = np.random.default_rng(0)
    updates = [rng.normal(0, 1, 10) for _ in range(6)]
    agg = SecureAggregator(seed=1)
    masked = agg.mask(updates)
    # individual masked updates must differ from raw (privacy)
    for u, m in zip(updates, masked):
        assert not np.allclose(u, m)
    recovered = agg.aggregate(masked)
    assert np.allclose(recovered, np.sum(np.vstack(updates), axis=0), atol=1e-9)


def test_single_client_mask_is_zero():
    masks = pairwise_masks(1, 5, seed=0)
    assert np.allclose(masks[0], 0.0)
    agg = SecureAggregator()
    u = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    assert np.allclose(agg.run(u), u[0])


def test_dimension_mismatch_raises():
    agg = SecureAggregator()
    import pytest

    with pytest.raises(ValueError):
        agg.mask([np.zeros(3), np.zeros(4)])
