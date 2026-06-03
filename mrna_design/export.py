"""
export.py — Excel export utilities for mRNA Library Designer.

Provides helpers to export library data as multi-sheet Excel workbooks
using in-memory BytesIO buffers suitable for Streamlit downloads.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd


def generate_timestamp() -> str:
    """Return a deterministic timestamp string for filenames (UTC, compact)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def export_library_excel(
    full_library_df: pd.DataFrame,
    best_candidates_df: pd.DataFrame | None = None,
    tie_band_df: pd.DataFrame | None = None,
    pareto_df: pd.DataFrame | None = None,
) -> io.BytesIO:
    """
    Export library data as a single Excel workbook with multiple sheets.

    Sheets included:
      - full_library (always)
      - best_candidates (if provided and non-empty)
      - tie_band (if provided and non-empty)
      - pareto_front (if provided and non-empty)

    Parameters
    ----------
    full_library_df : pd.DataFrame
        The complete library table.
    best_candidates_df : pd.DataFrame | None
        Shortlisted best candidates.
    tie_band_df : pd.DataFrame | None
        All candidates within the tie band.
    pareto_df : pd.DataFrame | None
        Pareto front candidates.

    Returns
    -------
    io.BytesIO
        In-memory buffer containing the .xlsx workbook.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        full_library_df.to_excel(writer, sheet_name="full_library", index=False)

        if best_candidates_df is not None and not best_candidates_df.empty:
            best_candidates_df.to_excel(
                writer, sheet_name="best_candidates", index=False
            )

        if tie_band_df is not None and not tie_band_df.empty:
            tie_band_df.to_excel(writer, sheet_name="tie_band", index=False)

        if pareto_df is not None and not pareto_df.empty:
            pareto_df.to_excel(writer, sheet_name="pareto_front", index=False)

    buf.seek(0)
    return buf


def export_best_candidates_excel(
    best_candidates_df: pd.DataFrame,
) -> io.BytesIO:
    """Export best candidates as a standalone Excel file."""
    buf = io.BytesIO()
    best_candidates_df.to_excel(buf, engine="openpyxl", index=False)
    buf.seek(0)
    return buf
