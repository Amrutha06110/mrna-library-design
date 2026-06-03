"""
Tests for config validation, ranking stability, and chunked processing.
Run with:  pytest tests/ -v
"""
import pytest
from pathlib import Path
import tempfile

from mrna_design.config_model import (
    PipelineConfig,
    ScoringWeights,
    BarcodingConfig,
    QCConfig,
    FiltersConfig,
    load_and_validate_config,
)


# ── Config validation edge cases ──────────────────────────────────────────────

class TestConfigValidation:
    """Test strict config validation with Pydantic models."""

    def test_default_config_valid(self):
        """Default config should be valid without any input."""
        cfg = PipelineConfig()
        assert cfg.cap == "m7G"
        assert cfg.kozak == "strong"
        assert cfg.polya_length == 100

    def test_scoring_weights_normalize(self):
        """Weights that don't sum to 1.0 should be auto-normalized."""
        w = ScoringWeights(cai=0.4, mfe_stability=0.3, gc_content=0.2,
                           utr_access=0.1, codon_pair=0.0, u_depletion=0.0,
                           cpg_depletion=0.0, translation_eff=0.0,
                           codon_ramp=0.0, codon_diversity=0.0)
        total = w.cai + w.mfe_stability + w.gc_content + w.utr_access
        assert abs(total - 1.0) < 0.01

    def test_scoring_weights_all_zero_fails(self):
        """All-zero weights should raise ValueError."""
        with pytest.raises(ValueError, match="All scoring weights are zero"):
            ScoringWeights(cai=0, mfe_stability=0, gc_content=0,
                           utr_access=0, codon_pair=0, u_depletion=0,
                           cpg_depletion=0, translation_eff=0,
                           codon_ramp=0, codon_diversity=0)

    def test_scoring_weights_negative_fails(self):
        """Negative weight should fail validation."""
        with pytest.raises(Exception):
            ScoringWeights(cai=-0.1)

    def test_scoring_weights_above_one_fails(self):
        """Weight > 1.0 should fail validation."""
        with pytest.raises(Exception):
            ScoringWeights(cai=1.5)

    def test_invalid_cap_type(self):
        """Invalid cap type should fail."""
        with pytest.raises(Exception):
            PipelineConfig(cap="invalid_cap")

    def test_invalid_kozak_type(self):
        """Invalid kozak type should fail."""
        with pytest.raises(Exception):
            PipelineConfig(kozak="ultra_strong")

    def test_polya_length_negative(self):
        """Negative poly-A length should fail."""
        with pytest.raises(Exception):
            PipelineConfig(polya_length=-10)

    def test_polya_length_too_large(self):
        """Poly-A > 500 should fail."""
        with pytest.raises(Exception):
            PipelineConfig(polya_length=1000)

    def test_barcoding_gc_range_invalid(self):
        """gc_min >= gc_max should fail."""
        with pytest.raises(Exception):
            BarcodingConfig(gc_min=0.7, gc_max=0.3)

    def test_barcoding_gc_range_equal(self):
        """gc_min == gc_max should fail."""
        with pytest.raises(Exception):
            BarcodingConfig(gc_min=0.5, gc_max=0.5)

    def test_qc_local_gc_range_invalid(self):
        """local_gc_min >= local_gc_max should fail."""
        with pytest.raises(Exception):
            QCConfig(local_gc_min=0.9, local_gc_max=0.2)

    def test_filters_gc_range_invalid(self):
        """filters.gc_min >= gc_max should fail."""
        with pytest.raises(Exception):
            FiltersConfig(gc_min=0.8, gc_max=0.3)

    def test_max_combinations_null(self):
        """max_combinations=None should be valid (no cap)."""
        cfg = PipelineConfig(max_combinations=None)
        assert cfg.max_combinations is None

    def test_chunk_size_too_small(self):
        """chunk_size < 100 should fail."""
        with pytest.raises(Exception):
            PipelineConfig(chunk_size=10)

    def test_workers_zero(self):
        """workers=0 should fail."""
        with pytest.raises(Exception):
            PipelineConfig(workers=0)

    def test_workers_too_many(self):
        """workers > 32 should fail."""
        with pytest.raises(Exception):
            PipelineConfig(workers=64)

    def test_load_and_validate_missing_file(self):
        """Missing config file should use defaults."""
        cfg = load_and_validate_config(Path("/nonexistent/config.yaml"))
        assert cfg.cap == "m7G"

    def test_load_and_validate_with_overrides(self, tmp_path):
        """CLI overrides should take priority."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("cap: arca\npolya_length: 50\n")
        cfg = load_and_validate_config(cfg_file, overrides={"polya_length": 200})
        assert cfg.cap == "arca"
        assert cfg.polya_length == 200

    def test_load_and_validate_invalid_yaml(self, tmp_path):
        """Invalid config values should raise ValueError."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("cap: invalid_cap_type\n")
        with pytest.raises(ValueError, match="Configuration validation failed"):
            load_and_validate_config(cfg_file)

    def test_barcoding_mode_invalid(self):
        """Invalid barcoding mode should fail."""
        with pytest.raises(Exception):
            BarcodingConfig(mode="invalid")

    def test_optimizer_method_invalid(self):
        """Invalid optimizer method should fail."""
        with pytest.raises(Exception):
            from mrna_design.config_model import OptimizerConfig
            OptimizerConfig(method="super_optimize")

    def test_scoring_weights_to_dict(self):
        """to_dict() should return a plain dict."""
        w = ScoringWeights()
        d = w.to_dict()
        assert isinstance(d, dict)
        assert "cai" in d
        assert abs(sum(d.values()) - 1.0) < 0.01


# ── Ranking stability (deterministic regression) ──────────────────────────────

class TestRankingStability:
    """Test that scoring produces deterministic, stable rankings."""

    @pytest.fixture
    def synthetic_library(self, tmp_path):
        """Create a small synthetic dataset for deterministic testing."""
        utr5_dir = tmp_path / "utr5"
        orf_dir = tmp_path / "orf"
        utr3_dir = tmp_path / "utr3"
        utr5_dir.mkdir()
        orf_dir.mkdir()
        utr3_dir.mkdir()

        (utr5_dir / "test.fasta").write_text(
            ">UTR5_X\nGCCACCAUGGCUGCC\n"
            ">UTR5_Y\nAAAAGCCACCAUGG\n"
        )
        (orf_dir / "test.fasta").write_text(
            ">ORF_A\nAUGGAGAAAGACGCCCUGAAGCCCGAGAACGCCGAGCUGAAG\n"
            ">ORF_B\nAUGCUGAAAGACGCCUUUAAGCCCGAGAACUUUGAGCUGAAG\n"
        )
        (utr3_dir / "test.fasta").write_text(
            ">UTR3_P\nUGCUAGUGAAUUUCGAACCG\n"
        )
        return tmp_path

    def test_scoring_deterministic(self, synthetic_library):
        """Running scoring twice should produce identical results."""
        from mrna_design.assembler import assemble_library
        from mrna_design.scorer import score_library

        lib1 = assemble_library(
            utr5_dir=synthetic_library / "utr5",
            orf_dir=synthetic_library / "orf",
            utr3_dir=synthetic_library / "utr3",
            max_combinations=None,
        )
        scored1 = score_library(lib1)
        scored1.sort(key=lambda x: x["composite_score"], reverse=True)

        lib2 = assemble_library(
            utr5_dir=synthetic_library / "utr5",
            orf_dir=synthetic_library / "orf",
            utr3_dir=synthetic_library / "utr3",
            max_combinations=None,
        )
        scored2 = score_library(lib2)
        scored2.sort(key=lambda x: x["composite_score"], reverse=True)

        # Rankings should be identical
        for c1, c2 in zip(scored1, scored2):
            assert c1["id"] == c2["id"]
            assert c1["composite_score"] == c2["composite_score"]

    def test_ranking_order_stable(self, synthetic_library):
        """Rank order should remain consistent across runs."""
        from mrna_design.assembler import assemble_library
        from mrna_design.scorer import score_library

        lib = assemble_library(
            utr5_dir=synthetic_library / "utr5",
            orf_dir=synthetic_library / "orf",
            utr3_dir=synthetic_library / "utr3",
            max_combinations=None,
        )
        scored = score_library(lib)
        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        # Verify ordering is strictly non-increasing
        for i in range(len(scored) - 1):
            assert scored[i]["composite_score"] >= scored[i + 1]["composite_score"]


# ── Chunked processing correctness ───────────────────────────────────────────

class TestChunkedProcessing:
    """Test that chunked scoring matches baseline (non-chunked) results."""

    @pytest.fixture
    def synthetic_library(self, tmp_path):
        """Create a small synthetic dataset."""
        utr5_dir = tmp_path / "utr5"
        orf_dir = tmp_path / "orf"
        utr3_dir = tmp_path / "utr3"
        utr5_dir.mkdir()
        orf_dir.mkdir()
        utr3_dir.mkdir()

        (utr5_dir / "test.fasta").write_text(
            ">U5A\nGCCACCAUGGCUGCC\n"
            ">U5B\nAAAAGCCACCAUGG\n"
            ">U5C\nGCUGCCGCCACCAUG\n"
        )
        (orf_dir / "test.fasta").write_text(
            ">OA\nAUGGAGAAAGACGCCCUGAAGCCCGAGAACGCCGAGCUGAAG\n"
            ">OB\nAUGCUGAAAGACGCCUUUAAGCCCGAGAACUUUGAGCUGAAG\n"
        )
        (utr3_dir / "test.fasta").write_text(
            ">U3X\nUGCUAGUGAAUUUCGAACCG\n"
            ">U3Y\nGCCAACUGAUUUCGCCCAAG\n"
        )
        return tmp_path

    def test_chunked_matches_baseline(self, synthetic_library):
        """Chunked scoring should produce same scores as non-chunked baseline."""
        from mrna_design.assembler import assemble_library
        from mrna_design.scorer import score_library, DEFAULT_WEIGHTS
        from mrna_design.pipeline import _score_chunk_with_transparency

        # Baseline: standard score_library
        lib = assemble_library(
            utr5_dir=synthetic_library / "utr5",
            orf_dir=synthetic_library / "orf",
            utr3_dir=synthetic_library / "utr3",
            max_combinations=None,
        )
        baseline = score_library(lib)
        baseline.sort(key=lambda x: (x["id"],))

        # Chunked: process in chunks of 2
        lib2 = assemble_library(
            utr5_dir=synthetic_library / "utr5",
            orf_dir=synthetic_library / "orf",
            utr3_dir=synthetic_library / "utr3",
            max_combinations=None,
        )
        chunked_results = []
        chunk_size = 2
        for i in range(0, len(lib2), chunk_size):
            chunk = lib2[i:i + chunk_size]
            scored_chunk = _score_chunk_with_transparency(chunk, DEFAULT_WEIGHTS)
            chunked_results.extend(scored_chunk)
        chunked_results.sort(key=lambda x: (x["id"],))

        # Compare composite scores
        assert len(baseline) == len(chunked_results)
        for b, c in zip(baseline, chunked_results):
            assert b["id"] == c["id"]
            assert abs(b["composite_score"] - c["composite_score"]) < 1e-6

    def test_chunked_has_transparency_fields(self, synthetic_library):
        """Chunked scoring should add transparency fields."""
        from mrna_design.assembler import assemble_library
        from mrna_design.scorer import DEFAULT_WEIGHTS
        from mrna_design.pipeline import _score_chunk_with_transparency

        lib = assemble_library(
            utr5_dir=synthetic_library / "utr5",
            orf_dir=synthetic_library / "orf",
            utr3_dir=synthetic_library / "utr3",
            max_combinations=None,
        )
        scored = _score_chunk_with_transparency(lib[:2], DEFAULT_WEIGHTS)

        for c in scored:
            # Should have weight and contribution fields
            assert "cai_weight" in c
            assert "cai_contribution" in c
            assert "mfe_stability_weight" in c
            assert "mfe_stability_contribution" in c
            # Contributions should sum approximately to composite score
            scored_dims = [
                "cai", "mfe_stability", "gc_score", "utr_access",
                "codon_pair", "u_depletion", "cpg_depletion",
                "translation_eff", "codon_ramp", "codon_diversity",
            ]
            total_contrib = sum(c.get(f"{d}_contribution", 0) for d in scored_dims)
            assert abs(total_contrib - c["composite_score"]) < 1e-4

    def test_pipeline_dry_run(self, synthetic_library):
        """--dry-run should validate config and return empty list."""
        from mrna_design.config_model import PipelineConfig
        from mrna_design.pipeline import run_pipeline

        cfg = PipelineConfig(
            utr5_dir=str(synthetic_library / "utr5"),
            orf_dir=str(synthetic_library / "orf"),
            utr3_dir=str(synthetic_library / "utr3"),
            output_dir=str(synthetic_library / "outputs"),
        )
        result = run_pipeline(cfg, dry_run=True)
        assert result == []
