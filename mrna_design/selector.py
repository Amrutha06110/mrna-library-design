"""
selector.py — Robust candidate selection with tie-banding and secondary tie-breakers.

Also provides a simple Pareto-front helper for multi-objective optimization.
"""
from __future__ import annotations

import pandas as pd


def select_best_candidates(
    df: pd.DataFrame,
    *,
    score_col: str = "Total_Score",
    epsilon: float = 0.0003,
    top_percentile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select best candidates using tie-banding and secondary tie-breakers.

    Parameters
    ----------
    df : pd.DataFrame
        Library dataframe. Must contain *score_col*.
    score_col : str
        Column with the total/composite score.
    epsilon : float
        Absolute tolerance for the tie band (default 0.0003).
    top_percentile : float | None
        If provided (e.g. 0.10 for top 10%), use percentile-based band instead
        of epsilon.

    Returns
    -------
    best_df : pd.DataFrame
        Final shortlisted candidates (sorted by tie-breakers).
    tie_band_df : pd.DataFrame
        All candidates within the tie band (before tie-breaking sort).
    """
    if df.empty or score_col not in df.columns:
        empty = df.iloc[0:0].copy()
        return empty, empty

    max_total = df[score_col].max()

    # Determine threshold
    if top_percentile is not None:
        threshold = df[score_col].quantile(1.0 - top_percentile)
    else:
        threshold = max_total - epsilon

    # Filter tie band
    tie_band_df = df[df[score_col] >= threshold].copy()

    # Apply secondary tie-breakers
    sort_keys: list[str] = []
    sort_ascending: list[bool] = []

    if "Score_MFE" in tie_band_df.columns:
        sort_keys.append("Score_MFE")
        sort_ascending.append(False)
    if "Score_GC" in tie_band_df.columns:
        sort_keys.append("Score_GC")
        sort_ascending.append(False)
    if "Name" in tie_band_df.columns:
        sort_keys.append("Name")
        sort_ascending.append(True)

    if sort_keys:
        best_df = tie_band_df.sort_values(
            by=sort_keys, ascending=sort_ascending
        ).reset_index(drop=True)
    else:
        best_df = tie_band_df.sort_values(
            by=score_col, ascending=False
        ).reset_index(drop=True)

    tie_band_df = tie_band_df.reset_index(drop=True)
    return best_df, tie_band_df


def pareto_front(
    df: pd.DataFrame,
    objectives: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute the Pareto front for maximising the given objective columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    objectives : list[str] | None
        Columns to maximise. Defaults to Score_MFE, Score_GC, Score_UTR
        (only those present).

    Returns
    -------
    pd.DataFrame
        Rows on the Pareto front.
    """
    if objectives is None:
        objectives = [c for c in ["Score_MFE", "Score_GC", "Score_UTR"] if c in df.columns]

    if not objectives or df.empty:
        return df.iloc[0:0].copy()

    # Extract objective matrix
    vals = df[objectives].values
    n = len(vals)
    is_dominated = [False] * n

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # Check if j dominates i (all >= and at least one >)
            if all(vals[j] >= vals[i]) and any(vals[j] > vals[i]):
                is_dominated[i] = True
                break

    pareto_mask = [not d for d in is_dominated]
    return df[pareto_mask].reset_index(drop=True)
