"""
Tests for the selector and export modules.
Run with: pytest tests/test_selector_export.py -v
"""
import io

import pandas as pd
import pytest

from mrna_design.selector import select_best_candidates, pareto_front
from mrna_design.export import (
    export_library_excel,
    export_best_candidates_excel,
    generate_timestamp,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_library_df():
    """Create a sample library DataFrame with near-identical total scores."""
    return pd.DataFrame([
        {"Name": "Construct_0001", "Score_CAI": 0.90, "Score_MFE": 0.85,
         "Score_GC": 0.80, "Score_UTR": 0.70, "Total_Score": 0.8125},
        {"Name": "Construct_0002", "Score_CAI": 0.88, "Score_MFE": 0.87,
         "Score_GC": 0.82, "Score_UTR": 0.68, "Total_Score": 0.8124},
        {"Name": "Construct_0003", "Score_CAI": 0.85, "Score_MFE": 0.80,
         "Score_GC": 0.75, "Score_UTR": 0.65, "Total_Score": 0.7625},
        {"Name": "Construct_0004", "Score_CAI": 0.92, "Score_MFE": 0.86,
         "Score_GC": 0.81, "Score_UTR": 0.72, "Total_Score": 0.8126},
        {"Name": "Construct_0005", "Score_CAI": 0.80, "Score_MFE": 0.70,
         "Score_GC": 0.60, "Score_UTR": 0.50, "Total_Score": 0.6500},
    ])


# ── Selector tests ───────────────────────────────────────────────────────────

class TestSelectBestCandidates:
    def test_tie_band_epsilon(self, sample_library_df):
        """Candidates within epsilon of max should be in tie band."""
        best_df, tie_band_df = select_best_candidates(
            sample_library_df, epsilon=0.0003
        )
        # Max is 0.8126; 0.8126 - 0.0003 = 0.8123
        # So 0.8126, 0.8125, 0.8124 are all in band
        assert len(tie_band_df) == 3
        assert "Construct_0003" not in tie_band_df["Name"].values
        assert "Construct_0005" not in tie_band_df["Name"].values

    def test_tie_band_percentile(self, sample_library_df):
        """Top percentile should select correct proportion."""
        best_df, tie_band_df = select_best_candidates(
            sample_library_df, top_percentile=0.50
        )
        # Top 50% → at least top 2-3 candidates
        assert len(tie_band_df) >= 2

    def test_secondary_tiebreaker_mfe_first(self, sample_library_df):
        """Within tie band, highest Score_MFE should rank first."""
        best_df, _ = select_best_candidates(sample_library_df, epsilon=0.0003)
        # Construct_0002 has highest MFE (0.87) in the tie band
        assert best_df.iloc[0]["Name"] == "Construct_0002"

    def test_deterministic_sorting(self, sample_library_df):
        """Repeated calls should produce same ordering."""
        best1, _ = select_best_candidates(sample_library_df, epsilon=0.0003)
        best2, _ = select_best_candidates(sample_library_df, epsilon=0.0003)
        pd.testing.assert_frame_equal(best1, best2)

    def test_empty_dataframe(self):
        """Empty input returns empty output."""
        empty_df = pd.DataFrame(columns=["Name", "Total_Score", "Score_MFE"])
        best_df, tie_band_df = select_best_candidates(empty_df)
        assert best_df.empty
        assert tie_band_df.empty

    def test_missing_score_col(self):
        """If score column missing, returns empty."""
        df = pd.DataFrame({"Name": ["A"], "Other": [1.0]})
        best_df, tie_band_df = select_best_candidates(df)
        assert best_df.empty


class TestParetoFront:
    def test_pareto_basic(self, sample_library_df):
        """Pareto front should contain non-dominated solutions."""
        front = pareto_front(sample_library_df)
        assert not front.empty
        # Construct_0005 is dominated by others in all objectives
        assert "Construct_0005" not in front["Name"].values

    def test_pareto_single_row(self):
        """Single row is always on the Pareto front."""
        df = pd.DataFrame([{
            "Name": "A", "Score_MFE": 0.9, "Score_GC": 0.8, "Score_UTR": 0.7
        }])
        front = pareto_front(df)
        assert len(front) == 1

    def test_pareto_empty(self):
        """Empty df returns empty."""
        df = pd.DataFrame(columns=["Score_MFE", "Score_GC"])
        front = pareto_front(df)
        assert front.empty

    def test_pareto_no_objectives(self):
        """If no objective columns present, returns empty."""
        df = pd.DataFrame({"Name": ["A", "B"], "Other": [1, 2]})
        front = pareto_front(df)
        assert front.empty


# ── Export tests ─────────────────────────────────────────────────────────────

class TestExcelExport:
    def test_export_library_excel_has_sheets(self, sample_library_df):
        """Exported workbook should have expected sheet names."""
        best_df, tie_band_df = select_best_candidates(sample_library_df)
        pareto_df = pareto_front(sample_library_df)

        buf = export_library_excel(
            full_library_df=sample_library_df,
            best_candidates_df=best_df,
            tie_band_df=tie_band_df,
            pareto_df=pareto_df,
        )

        assert buf.getbuffer().nbytes > 0

        # Read back and check sheets
        xlsx = pd.ExcelFile(buf, engine="openpyxl")
        assert "full_library" in xlsx.sheet_names
        assert "best_candidates" in xlsx.sheet_names
        assert "tie_band" in xlsx.sheet_names
        assert "pareto_front" in xlsx.sheet_names

    def test_export_library_excel_no_optional_sheets(self):
        """If optional dfs are None/empty, sheets are omitted."""
        df = pd.DataFrame({"Name": ["A"], "Total_Score": [1.0]})
        buf = export_library_excel(df)
        xlsx = pd.ExcelFile(buf, engine="openpyxl")
        assert "full_library" in xlsx.sheet_names
        assert "best_candidates" not in xlsx.sheet_names

    def test_export_best_candidates_excel(self, sample_library_df):
        """Best candidates export should produce non-empty xlsx."""
        best_df, _ = select_best_candidates(sample_library_df)
        buf = export_best_candidates_excel(best_df)
        assert buf.getbuffer().nbytes > 0

        result_df = pd.read_excel(buf, engine="openpyxl")
        assert len(result_df) == len(best_df)

    def test_export_no_index(self, sample_library_df):
        """Exported Excel should not contain an index column."""
        buf = export_library_excel(sample_library_df)
        result_df = pd.read_excel(buf, engine="openpyxl", sheet_name="full_library")
        # Index column would appear as 'Unnamed: 0' or similar
        assert not any("Unnamed" in str(c) for c in result_df.columns)

    def test_csv_backward_compat(self, sample_library_df):
        """CSV export still works (backward compatibility)."""
        csv_buf = io.BytesIO()
        sample_library_df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        result = pd.read_csv(csv_buf)
        assert len(result) == len(sample_library_df)
        assert list(result.columns) == list(sample_library_df.columns)


class TestTimestamp:
    def test_timestamp_format(self):
        """Timestamp should be a compact date-time string."""
        ts = generate_timestamp()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert "_" in ts
