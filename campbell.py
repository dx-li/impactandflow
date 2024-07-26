import numpy as np


def estimate_effs(metaorder_sizes: np.ndarray, decision_prices: np.ndarray, N: int = 30, b: int = 4, bucket_func = None):
    """
    Estimate the Expected Future Flow Shortfall (EFFS) using historical data.

    :param metaorder_sizes: Array of metaorder sizes
    :type metaorder_sizes: np.ndarray
    :param decision_prices: Array of decision prices
    :type decision_prices: np.ndarray
    :param N: Number of future flows to consider
    :type N: int
    :param b: Number of buckets to split the range of metaorder sizes into
    :type b: int
    :param bucket_func: Function to split the range of metaorder sizes into buckets
    :type bucket_func: Callable(np.ndarray, int) -> np.ndarray
    :return: Estimated EFFS for each metaorder
    :rtype: np.ndarray
    """
    
    eff = estimate_eff(metaorder_sizes, N, 2, bucket_func)
    price_delta = np.empty(len(eff))
    price_delta[:] = np.nan
    price_delta[:-1] = decision_prices[1:] - decision_prices[:-1]
    return eff * price_delta


def estimate_eff(metaorder_sizes: np.ndarray, N: int, b: int, bucket_func = None):
    """
    Estimate the Expected Future Flow (EFF) using historical data.
    
    :param metaorder_sizes: Array of metaorder sizes
    :type metaorder_sizes: np.ndarray
    :param N: Number of future flows to consider
    :type N: int
    :param b: Number of buckets to split the range of metaorder sizes into
    :type b: int
    :param bucket_func: Function to split the range of metaorder sizes into buckets
    :type bucket_func: Callable(np.ndarray, int) -> np.ndarray
    :return: Estimated EFF for each metaorder
    :rtype: np.ndarray
    """
    
    if bucket_func is None:
        bucket_func = quantile_buckets
    # Step 0: Separate sign and magnitude
    metaorder_sides = np.sign(metaorder_sizes)
    metaorder_sizes_abs = np.abs(metaorder_sizes)

    # Step 1: Split the range of metaorder sizes into buckets
    bucketed = bucket_func(metaorder_sizes_abs, b)

    # Step 2: Compute side-adjusted future flows
    adjusted_flows = side_adjusted_future_flows(bucketed, metaorder_sides, metaorder_sizes, N)
    # Step 3: Compute mean flow for each bucket and restore the side
    means = {}
    for bucket in np.unique(bucketed):
        idx = bucketed == bucket
        means[bucket] = np.nanmean(adjusted_flows[idx])
    
    estimated_effs = np.array([means[bucket] for bucket in bucketed]) * metaorder_sides

    return estimated_effs

def quantile_buckets(metaorder_sizes, num_buckets):
    T = len(metaorder_sizes)
    nq = num_buckets + 1
    o = metaorder_sizes.argpartition(np.arange(1, nq) * T // nq)
    out = np.empty(T, int)
    out[o] = np.arange(T) * nq // T
    return out

def side_adjusted_future_flows(bucketed, metaorder_sides, metaorder_sizes, N):
    sum_flows = np.empty(len(bucketed))
    for i in range(1, len(bucketed)-N+1):
        sum_flows[i-1] = np.sum(metaorder_sizes[i:i+N])
    adjusted_flows = metaorder_sides * sum_flows
    return adjusted_flows

