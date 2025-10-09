# -*- coding: utf-8 -*-
# 条件付き CDF: 離散=経験CDF, 連続=カーネル推定
import re, math
import numpy as np
import pandas as pd

def wide_to_long_bv(df_wide: pd.DataFrame) -> pd.DataFrame:
    """列名末尾の数値を v とみなし、(b,v) の縦持ちに変換"""
    records = []
    for col in df_wide.columns:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(col))
        if not nums: 
            continue
        v_val = float(nums[-1])
        b_vals = pd.to_numeric(df_wide[col], errors="coerce").dropna().to_numpy()
        records.extend([(float(b), v_val) for b in b_vals])
    return pd.DataFrame(records, columns=["b", "v"]).dropna().reset_index(drop=True)

def conditional_empirical_cdf_discrete(df_long: pd.DataFrame, v0: float,
                                       tol: float = 0.0, min_n: int = 1):
    """
    離散: 同一(±tol)の v だけで経験CDFを返す。
    Returns: b_grid, cdf_vals (np.ndarray)
    """
    arr = df_long[["b", "v"]].to_numpy(float)
    b, v = arr[:, 0], arr[:, 1]
    mask = (np.abs(v - v0) <= tol) if tol > 0 else (v == v0)
    b_sub = b[mask]; n = b_sub.size
    if n < min_n:
        return np.array([]), np.array([])
    b_sorted = np.sort(b_sub)
    b_grid  = np.unique(b_sorted)
    counts  = np.searchsorted(b_sorted, b_grid, side="right")
    cdf     = np.clip(counts / n, 0.0, 1.0)
    return b_grid, cdf

def _gaussian_kernel(u):
    return np.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)

def _silverman_bandwidth(x):
    x = np.asarray(x); n = x.size
    if n < 2: return 1.0
    std = np.nanstd(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    s = min(std, iqr/1.349) if (std>0 and iqr>0) else max(std, iqr/1.349, 1e-8)
    return 1.06 * s * (n ** (-1/5))

def conditional_cdf_kernel(df_long: pd.DataFrame, v0: float,
                           b_points=None, bandwidth: float | None = None):
    """
    連続: Nadaraya–Watson型  F(b|v0)=Σ K((v_i-v0)/h)1{b_i<=b}/ΣK((v_i-v0)/h)
    Returns: b_grid, cdf_est (np.ndarray)
    """
    arr = df_long[["b", "v"]].to_numpy(float)
    b, v = arr[:, 0], arr[:, 1]
    if b.size == 0:
        return np.array([]), np.array([])
    if b_points is None:
        bmin, bmax = float(np.min(b)), float(np.max(b))
        if bmin == bmax: bmin -= 1.0; bmax += 1.0
        b_points = np.linspace(bmin, bmax, 200)
    else:
        b_points = np.asarray(b_points)
    if bandwidth is None or bandwidth <= 0:
        bandwidth = _silverman_bandwidth(v)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = max(np.std(v), 1e-6)
    w = _gaussian_kernel((v - v0)/bandwidth)
    denom = w.sum()
    if denom <= 1e-12:  # 近傍データが極端に少ない場合のフォールバック
        w = np.ones_like(w); denom = w.size
    indicator = (b[:, None] <= b_points[None, :]).astype(float)
    cdf = (w[:, None] * indicator).sum(axis=0) / denom
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    return b_points, cdf
