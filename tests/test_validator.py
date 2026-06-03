"""
Tests for mrna_design.validator module.

Covers passing, failing, and edge-case scenarios for all six validators.
Run with:  pytest tests/test_validator.py -v
"""
import math
import pytest

from mrna_design.validator import (
    validate_gc_bounds,
    validate_kozak_consensus,
    validate_homopolymer_dinuc,
    validate_mfe_sanity,
    validate_score_distribution,
    validate_reproducibility,
    run_all_validations,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. validate_gc_bounds
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateGCBounds:
    """Tests for GC content validation."""

    def test_passing_50_percent(self):
        # 50% GC sequence
        seq = "GCGCAUAU" * 10  # 50% GC
        result = validate_gc_bounds(seq)
        assert result["passed"] is True
        assert result["check_name"] == "gc_bounds"
        assert result["severity"] == "info"

    def test_passing_at_lower_bound(self):
        # Exactly 40% GC
        seq = "GC" * 20 + "AAUU" * 15  # 40/(40+60) = 40%
        result = validate_gc_bounds(seq, gc_min=0.40, gc_max=0.60)
        gc = result["details"]["gc_fraction"]
        assert 0.39 <= gc <= 0.41
        assert result["passed"] is True

    def test_failing_too_low(self):
        # Very low GC
        seq = "AAAUUUAAAUUU" * 10  # 0% GC
        result = validate_gc_bounds(seq)
        assert result["passed"] is False
        assert result["details"]["gc_fraction"] == 0.0

    def test_failing_too_high(self):
        # Very high GC
        seq = "GCGCGCGCGCGCGCGC" * 5  # 100% GC
        result = validate_gc_bounds(seq)
        assert result["passed"] is False
        assert result["details"]["gc_fraction"] == 1.0

    def test_custom_bounds(self):
        seq = "GCGCAAAA"  # 50% GC
        result = validate_gc_bounds(seq, gc_min=0.60, gc_max=0.80)
        assert result["passed"] is False

    def test_empty_sequence(self):
        result = validate_gc_bounds("")
        assert result["passed"] is False
        assert result["severity"] == "critical"

    def test_single_nucleotide(self):
        result = validate_gc_bounds("G")
        assert result["details"]["gc_fraction"] == 1.0

    def test_case_insensitive(self):
        r1 = validate_gc_bounds("GCGCauau")
        r2 = validate_gc_bounds("GCGCAUAU")
        assert r1["details"]["gc_fraction"] == r2["details"]["gc_fraction"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. validate_kozak_consensus
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateKozakConsensus:
    """Tests for Kozak consensus validation."""

    def test_perfect_kozak_a(self):
        # GCC A CC + AUGG
        utr5 = "XXXXXGCCACC"
        orf = "AUGG"
        result = validate_kozak_consensus(utr5, orf)
        assert result["passed"] is True

    def test_perfect_kozak_g(self):
        # GCC G CC + AUGG
        utr5 = "XXXXXGCCGCC"
        orf = "AUGG"
        result = validate_kozak_consensus(utr5, orf)
        assert result["passed"] is True

    def test_failing_no_kozak(self):
        utr5 = "AAAAAAAAUUU"
        orf = "AUGCCC"
        result = validate_kozak_consensus(utr5, orf)
        assert result["passed"] is False
        assert result["severity"] == "warning"

    def test_kozak_in_utr5_only(self):
        # Full Kozak pattern within UTR5 (ends with AUGG)
        utr5 = "AAAGCCACCAUGG"
        result = validate_kozak_consensus(utr5, "")
        assert result["passed"] is True

    def test_short_utr5(self):
        utr5 = "ACC"
        orf = "AUGG"
        result = validate_kozak_consensus(utr5, orf)
        # Too short to have full Kozak
        assert result["check_name"] == "kozak_consensus"

    def test_empty_inputs(self):
        result = validate_kozak_consensus("", "")
        assert result["passed"] is False

    def test_t_to_u_conversion(self):
        # Should work with T as well (DNA input)
        utr5 = "XXXXXGCCACC"
        orf = "ATGG"
        result = validate_kozak_consensus(utr5, orf)
        assert result["passed"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. validate_homopolymer_dinuc
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateHomopolymerDinuc:
    """Tests for homopolymer and dinucleotide repeat validation."""

    def test_passing_no_repeats(self):
        seq = "AUGCAUGCAUGCAUGC"
        result = validate_homopolymer_dinuc(seq)
        assert result["passed"] is True

    def test_failing_homopolymer_5(self):
        # Run of 5 A's — exactly at threshold (max_homopolymer=4 means >4 fails)
        seq = "AUGCAAAAACUGC"
        result = validate_homopolymer_dinuc(seq, max_homopolymer=4)
        assert result["passed"] is False
        assert len(result["details"]["homopolymer_violations"]) > 0

    def test_passing_homopolymer_at_limit(self):
        # Run of 4 A's — exactly at limit
        seq = "AUGCAAAACUGC"
        result = validate_homopolymer_dinuc(seq, max_homopolymer=4)
        assert result["passed"] is True

    def test_failing_dinucleotide_repeats(self):
        # 8 repeats of AC
        seq = "AUGC" + "AC" * 8 + "UGCA"
        result = validate_homopolymer_dinuc(seq, max_dinuc_repeat=6)
        assert result["passed"] is False
        assert len(result["details"]["dinuc_violations"]) > 0

    def test_passing_dinuc_at_limit(self):
        # 6 repeats of AC — exactly at limit
        seq = "AUGC" + "AC" * 6 + "UGCA"
        result = validate_homopolymer_dinuc(seq, max_dinuc_repeat=6)
        assert result["passed"] is True

    def test_empty_sequence(self):
        result = validate_homopolymer_dinuc("")
        assert result["passed"] is False
        assert result["severity"] == "critical"

    def test_custom_thresholds(self):
        seq = "AAAAAA"  # 6 A's
        # With max_homopolymer=6, this should pass (not >6)
        result = validate_homopolymer_dinuc(seq, max_homopolymer=6)
        assert result["passed"] is True
        # With max_homopolymer=5, this should fail
        result = validate_homopolymer_dinuc(seq, max_homopolymer=5)
        assert result["passed"] is False

    def test_multiple_violations(self):
        seq = "AAAAA" + "GCGC" + "CCCCC"
        result = validate_homopolymer_dinuc(seq, max_homopolymer=4)
        assert result["passed"] is False
        assert len(result["details"]["homopolymer_violations"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. validate_mfe_sanity
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateMFESanity:
    """Tests for MFE plausibility validation."""

    def test_passing_normal_mfe(self):
        # -0.3 per nt is typical for mRNA
        result = validate_mfe_sanity(mfe=-150.0, sequence_length=500)
        assert result["passed"] is True
        assert result["severity"] == "info"

    def test_failing_positive_mfe(self):
        result = validate_mfe_sanity(mfe=5.0, sequence_length=100)
        assert result["passed"] is False
        assert result["severity"] == "critical"

    def test_failing_too_negative(self):
        # -0.8 per nt is unrealistically stable
        result = validate_mfe_sanity(mfe=-400.0, sequence_length=500)
        assert result["passed"] is False
        assert "too negative" in result["message"].lower() or "overly stable" in result["message"].lower()

    def test_failing_not_negative_enough(self):
        # -0.005 per nt is suspiciously unstructured
        result = validate_mfe_sanity(mfe=-2.5, sequence_length=500)
        assert result["passed"] is False

    def test_zero_length(self):
        result = validate_mfe_sanity(mfe=-10.0, sequence_length=0)
        assert result["passed"] is False
        assert result["severity"] == "critical"

    def test_boundary_values(self):
        # Exactly at min boundary: -0.6 per nt
        result = validate_mfe_sanity(mfe=-60.0, sequence_length=100)
        assert result["passed"] is True

        # Exactly at max boundary: -0.01 per nt
        result = validate_mfe_sanity(mfe=-1.0, sequence_length=100)
        assert result["passed"] is True

    def test_zero_mfe(self):
        result = validate_mfe_sanity(mfe=0.0, sequence_length=100)
        assert result["passed"] is False
        assert result["severity"] == "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. validate_score_distribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateScoreDistribution:
    """Tests for score distribution validation."""

    def test_passing_varied_scores(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9, 0.4, 0.6, 0.8, 0.2, 0.45]
        result = validate_score_distribution(scores)
        assert result["passed"] is True

    def test_failing_all_identical(self):
        scores = [0.5] * 20
        result = validate_score_distribution(scores)
        assert result["passed"] is False
        assert result["severity"] == "critical"

    def test_failing_near_identical(self):
        scores = [0.500000 + i * 0.0000001 for i in range(20)]
        result = validate_score_distribution(scores)
        assert result["passed"] is False

    def test_empty_scores(self):
        result = validate_score_distribution([])
        assert result["passed"] is False
        assert result["severity"] == "critical"

    def test_single_score(self):
        result = validate_score_distribution([0.75])
        assert result["passed"] is True  # Not applicable for single score

    def test_two_scores_different(self):
        result = validate_score_distribution([0.3, 0.8])
        assert result["passed"] is True

    def test_custom_thresholds(self):
        scores = [0.5, 0.501, 0.502, 0.503]
        # With very low threshold, should pass
        result = validate_score_distribution(scores, min_std_dev=0.0001)
        assert result["passed"] is True

    def test_details_populated(self):
        scores = [0.1, 0.5, 0.9]
        result = validate_score_distribution(scores)
        assert "mean" in result["details"]
        assert "std_dev" in result["details"]
        assert "unique_count" in result["details"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. validate_reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateReproducibility:
    """Tests for reproducibility validation."""

    def test_passing_deterministic_fn(self):
        import random

        def deterministic_pipeline(seed=42):
            random.seed(seed)
            return [{"composite_score": random.random()} for _ in range(5)]

        result = validate_reproducibility(deterministic_pipeline, seed=42)
        assert result["passed"] is True

    def test_failing_nondeterministic_fn(self):
        import os

        call_count = {"n": 0}

        def nondeterministic_pipeline(seed=42):
            call_count["n"] += 1
            # Intentionally ignores seed
            return [{"composite_score": float(call_count["n"] + i)} for i in range(3)]

        result = validate_reproducibility(nondeterministic_pipeline, seed=42)
        assert result["passed"] is False

    def test_pipeline_raises_exception(self):
        def broken_pipeline(seed=42):
            raise ValueError("Something went wrong")

        result = validate_reproducibility(broken_pipeline, seed=42)
        assert result["passed"] is False
        assert result["severity"] == "critical"
        assert "failed" in result["message"].lower() or "error" in result["details"].get("error", "").lower()

    def test_different_output_lengths(self):
        call_count = {"n": 0}

        def varying_length_pipeline(seed=42):
            call_count["n"] += 1
            return [{"composite_score": 0.5}] * call_count["n"]

        result = validate_reproducibility(varying_length_pipeline, seed=42)
        assert result["passed"] is False

    def test_custom_seed(self):
        import random

        def pipeline(seed=0):
            random.seed(seed)
            return [{"composite_score": random.random()} for _ in range(3)]

        result = validate_reproducibility(pipeline, seed=123)
        assert result["passed"] is True
        assert result["details"]["seed"] == 123


# ═══════════════════════════════════════════════════════════════════════════════
# run_all_validations integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunAllValidations:
    """Integration tests for run_all_validations."""

    def test_basic_library(self):
        constructs = [
            {
                "full_sequence": "GCGCAUAUGCGCAUAU" * 10,  # ~50% GC
                "utr5_seq": "AAAGCCACCAUGG",
                "orf_seq": "AUGGGCGCAUAU",
                "mfe": -50.0,
                "composite_score": 0.75,
            },
            {
                "full_sequence": "AUGCAUGCAUGCAUGC" * 10,
                "utr5_seq": "AAAGCCGCCAUGG",
                "orf_seq": "AUGGCCCAAAU",
                "mfe": -45.0,
                "composite_score": 0.65,
            },
        ]
        results = run_all_validations(constructs)
        assert len(results) > 0
        assert all("check_name" in r for r in results)
        assert all("passed" in r for r in results)

    def test_selective_checks(self):
        constructs = [{"full_sequence": "GCGCAUAU" * 10, "composite_score": 0.5}]
        results = run_all_validations(
            constructs,
            checks_enabled={"gc_bounds": True, "kozak_consensus": False,
                          "homopolymer_dinuc": False, "mfe_sanity": False,
                          "score_distribution": False},
        )
        check_names = [r["check_name"] for r in results]
        assert "gc_bounds" in check_names
        assert "kozak_consensus" not in check_names

    def test_empty_library(self):
        results = run_all_validations([])
        # Should return no per-construct results, but score_distribution fails on empty
        assert any(r["check_name"] == "score_distribution" for r in results) is False
