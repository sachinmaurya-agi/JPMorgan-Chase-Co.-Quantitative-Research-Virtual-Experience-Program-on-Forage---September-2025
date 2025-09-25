
"""
quantization_rating_map.py

Utilities to quantize a 1D score (e.g., FICO) into a fixed number of buckets and produce a rating map:
 - Lower rating = better credit (rating 1 is best/highest FICO)
 - Several quantization strategies are provided:
    * equal_width_buckets
    * equal_frequency_buckets
    * kmeans_1d (weighted)
    * dp_optimal_mse (dynamic programming minimizing within-bucket MSE)
    * dp_optimal_likelihood (dynamic programming maximizing binomial log-likelihood)
 - make_rating_function(boundaries, invert=True): returns a function score->rating (1..K)
 - get_bucket_stats(scores, defaults, boundaries): returns counts and default rates per bucket

The DP methods operate on aggregated unique score values to be efficient. The cost functions are
computed using prefix sums and produce exact optimal solutions in O(K * U^2) time where U is the
number of unique score values and K the requested number of buckets. This is suitable for
moderate U (a few thousands).

Author: ChatGPT (utility)
"""

from typing import List, Tuple, Callable, Optional
import numpy as np
from collections import defaultdict

# --- helpers: aggregate unique values with counts ---
def _aggregate(scores: np.ndarray, defaults: np.ndarray):
    # returns arrays of unique sorted score values, counts n_i and defaults k_i aligned
    order = np.argsort(scores)
    s_sorted = scores[order]
    d_sorted = defaults[order]
    vals = []
    counts = []
    def_counts = []
    if s_sorted.size == 0:
        return np.array([]), np.array([]), np.array([])
    cur = s_sorted[0]
    cur_n = 0
    cur_k = 0
    for v,k in zip(s_sorted, d_sorted):
        if v == cur:
            cur_n += 1
            cur_k += int(k)
        else:
            vals.append(cur)
            counts.append(cur_n)
            def_counts.append(cur_k)
            cur = v
            cur_n = 1
            cur_k = int(k)
    vals.append(cur); counts.append(cur_n); def_counts.append(cur_k)
    return np.array(vals, dtype=float), np.array(counts, dtype=int), np.array(def_counts, dtype=int)

# --- equal width / equal frequency ---
def equal_width_buckets(scores: np.ndarray, n_buckets: int) -> List[float]:
    lo, hi = float(np.min(scores)), float(np.max(scores))
    edges = list(np.linspace(lo, hi, n_buckets+1))
    return edges

def equal_frequency_buckets(scores: np.ndarray, n_buckets: int) -> List[float]:
    quantiles = [i / n_buckets for i in range(n_buckets+1)]
    edges = list(np.quantile(scores, quantiles))
    return edges

# --- kmeans 1D using weighted unique values ---
def kmeans_1d(scores: np.ndarray, n_buckets: int, random_state: Optional[int]=0) -> List[float]:
    from sklearn.cluster import KMeans
    vals, counts, _ = _aggregate(scores, np.zeros_like(scores))
    vals = vals.reshape(-1,1)
    km = KMeans(n_clusters=n_buckets, random_state=random_state)
    km.fit(vals, sample_weight=counts)
    centers = np.sort(km.cluster_centers_.ravel())
    # boundaries are midpoints between sorted centers, with extremes extended
    edges = []
    # lower bound
    edges.append(float(np.min(scores)))
    for a,b in zip(centers[:-1], centers[1:]):
        edges.append(float((a+b)/2.0))
    edges.append(float(np.max(scores)))
    return edges

# --- dynamic programming helpers for cost computation ---
def _prefix_sums_for_vals(vals: np.ndarray, counts: np.ndarray, def_counts: np.ndarray):
    # vals: unique sorted score values
    w = counts.astype(float)
    s1 = (w * vals)
    s2 = (w * (vals**2))
    # prefix sums (1-indexed padding)
    n_pref = np.concatenate([[0.0], np.cumsum(w)])
    sum_pref = np.concatenate([[0.0], np.cumsum(s1)])
    sumsq_pref = np.concatenate([[0.0], np.cumsum(s2)])
    def_pref = np.concatenate([[0], np.cumsum(def_counts)])
    return n_pref, sum_pref, sumsq_pref, def_pref

def _interval_mse_cost(i: int, j: int, n_pref, sum_pref, sumsq_pref):
    # cost for interval [i..j] inclusive where i,j are 1-based indices into the unique-values arrays
    n = n_pref[j] - n_pref[i-1]
    if n <= 0:
        return 0.0
    s = sum_pref[j] - sum_pref[i-1]
    sq = sumsq_pref[j] - sumsq_pref[i-1]
    # sum (x - mean)^2 = sumsq - (sum^2)/n
    cost = sq - (s*s)/n
    return float(cost)

def _interval_negloglik_cost(i: int, j: int, n_pref, sum_pref, sumsq_pref, def_pref, eps=1e-9):
    # negative log-likelihood cost for binomial model on interval [i..j]
    n = int(n_pref[j] - n_pref[i-1])
    k = int(def_pref[j] - def_pref[i-1])
    # Laplace smoothing to avoid log(0)
    p = (k + 0.5) / (n + 1.0)
    # cost = - (k*log p + (n-k)*log(1-p))
    if n <= 0:
        return 0.0
    cost = - (k * np.log(p + eps) + (n - k) * np.log(1.0 - p + eps))
    return float(cost)

# --- DP: optimal partitioning into K buckets minimizing cost ---
def _dp_optimize(n_unique: int, K: int, cost_fn):
    # cost_fn(i,j) expects 1-based inclusive indices i<=j and returns a float cost
    INF = 1e18
    dp = [[INF] * (n_unique+1) for _ in range(K+1)]
    prev = [[-1] * (n_unique+1) for _ in range(K+1)]
    dp[0][0] = 0.0
    for k in range(1, K+1):
        for j in range(1, n_unique+1):
            best = INF
            best_i = -1
            # try placing last cut between i..j inclusive
            for i in range(1, j+1):
                val = dp[k-1][i-1] + cost_fn(i, j)
                if val < best:
                    best = val
                    best_i = i-1
            dp[k][j] = best
            prev[k][j] = best_i
    # reconstruct boundaries (indices in unique-values)
    boundaries_indices = []
    k = K
    j = n_unique
    while k > 0 and j > 0:
        i_minus_1 = prev[k][j]
        boundaries_indices.append((i_minus_1+1, j))  # interval start..end as 1-based inclusive
        j = i_minus_1
        k -= 1
    boundaries_indices.reverse()
    return boundaries_indices, dp

# --- public DP methods ---
def dp_optimal_mse(scores: np.ndarray, defaults: np.ndarray, n_buckets: int) -> List[float]:
    vals, counts, def_counts = _aggregate(scores, defaults)
    if vals.size == 0:
        return []
    n_pref, sum_pref, sumsq_pref, def_pref = _prefix_sums_for_vals(vals, counts, def_counts)
    n_unique = len(vals)
    def cost_fn(i, j):
        return _interval_mse_cost(i, j, n_pref, sum_pref, sumsq_pref)
    intervals, dp = _dp_optimize(n_unique, n_buckets, cost_fn)
    # convert intervals (based on unique vals) to boundaries: use min score and max score and interval edges midpoints
    edges = [float(vals[0])]
    for (_, end) in intervals[:-1]:
        # boundary between intervals at midpoint between vals[end-1] and vals[end]
        right_val = vals[end-1]
        next_val = vals[end]  # end is 1-based
        edges.append(float((right_val + next_val) / 2.0))
    edges.append(float(vals[-1]))
    return edges

def dp_optimal_likelihood(scores: np.ndarray, defaults: np.ndarray, n_buckets: int) -> List[float]:
    vals, counts, def_counts = _aggregate(scores, defaults)
    if vals.size == 0:
        return []
    n_pref, sum_pref, sumsq_pref, def_pref = _prefix_sums_for_vals(vals, counts, def_counts)
    n_unique = len(vals)
    def cost_fn(i, j):
        return _interval_negloglik_cost(i, j, n_pref, sum_pref, sumsq_pref, def_pref)
    intervals, dp = _dp_optimize(n_unique, n_buckets, cost_fn)
    edges = [float(vals[0])]
    for (_, end) in intervals[:-1]:
        right_val = vals[end-1]
        next_val = vals[end]
        edges.append(float((right_val + next_val) / 2.0))
    edges.append(float(vals[-1]))
    return edges

# --- utility to compute per-bucket stats given boundaries ---
def get_bucket_stats(scores: np.ndarray, defaults: np.ndarray, boundaries: List[float]):
    # boundaries is list of edges length K+1
    boundaries = list(boundaries)
    K = len(boundaries)-1
    idx = np.searchsorted(boundaries, scores, side='right') - 1
    # clamp idx
    idx = np.clip(idx, 0, K-1)
    stats = []
    for k in range(K):
        mask = (idx == k)
        n = int(mask.sum())
        kdef = int(np.sum(defaults[mask])) if n>0 else 0
        pd = (kdef / n) if n>0 else None
        stats.append({"bucket": k+1, "range": (boundaries[k], boundaries[k+1]), "n": n, "defaults": kdef, "pd": pd})
    return stats

# --- rating function: lower rating = better credit (1 best) ---
def make_rating_function(boundaries: List[float], invert: bool=True) -> Callable[[float], int]:
    # returns a function f(score) -> rating in 1..K
    b = list(boundaries)
    K = len(b)-1
    def f(score):
        # find which bucket
        # handle NaN
        if score is None or (isinstance(score, float) and np.isnan(score)):
            return None
        # searchsorted
        idx = np.searchsorted(b, score, side='right') - 1
        idx = int(np.clip(idx, 0, K-1))
        rating = idx + 1
        if invert:
            # invert so that highest score gets rating 1
            return int((K+1) - rating)
        else:
            return rating
    return f

# --- small convenience wrapper that chooses method and returns boundaries + rating func ---
def build_rating_map(scores: np.ndarray, defaults: np.ndarray, n_buckets: int,
                     method: str = "dp_mse", random_state: Optional[int]=0):
    method = method.lower()
    if method == "equal_width":
        edges = equal_width_buckets(scores, n_buckets)
    elif method == "equal_freq":
        edges = equal_frequency_buckets(scores, n_buckets)
    elif method == "kmeans":
        edges = kmeans_1d(scores, n_buckets, random_state=random_state)
    elif method == "dp_mse":
        edges = dp_optimal_mse(scores, defaults, n_buckets)
    elif method == "dp_ll" or method == "dp_likelihood" or method == "dp_loglik":
        edges = dp_optimal_likelihood(scores, defaults, n_buckets)
    else:
        raise ValueError("Unknown method: " + str(method))
    rating_fn = make_rating_function(edges, invert=True)
    stats = get_bucket_stats(scores, defaults, edges)
    return edges, rating_fn, stats

if __name__ == "__main__":
    # Quick demo using an optional CSV in the current folder named "Task 3 and 4_Loan_Data.csv"
    import pandas as pd, numpy as np
    csv_path = "Task 3 and 4_Loan_Data.csv"
    if not os.path.exists(csv_path):
        print("Demo CSV not found. Place Task 3 and 4_Loan_Data.csv next to this script to run demo.")
    else:
        df = pd.read_csv(csv_path)
        # attempt to find fico or fico_score column; fall back to 'fico_score' exact name
        possible = [c for c in df.columns if 'fico' in c.lower()]
        if not possible:
            raise SystemExit("No fico-like column found in CSV.")
        score_col = possible[0]
        # find target default column
        possible_targets = [c for c in df.columns if 'default' in c.lower()]
        if not possible_targets:
            raise SystemExit("No default-like column found in CSV.")
        target_col = possible_targets[0]
        print("Using score_col=", score_col, "target_col=", target_col)
        scores = df[score_col].astype(float).values
        defaults = df[target_col].apply(lambda x: 1 if x in [1,'1',True,'Y','y','Yes','yes'] else 0).values
        edges, rating_fn, stats = build_rating_map(scores, defaults, n_buckets=5, method='dp_ll')
        print("Boundaries (edges):", edges)
        print("Bucket stats:")
        for s in stats:
            print(s)
