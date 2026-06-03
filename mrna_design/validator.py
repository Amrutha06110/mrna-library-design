"""
validator.py — Validation module for construct quality checks and reproducibility testing.

Provides six validators that catch biologically implausible constructs or
scoring anomalies before export:

  - validate_gc_bounds: GC% within configurable range (default 40–60%)
  - validate_kozak_consensus: 5'UTR–ORF junction matches Kozak consensus
  - validate_homopolymer_dinuc: No homopolymer runs >4 or dinucleotide repeats >6
  - validate_mfe_sanity: MFE is negative and within plausible range
  - validate_score_distribution: Scores aren't all identical / std dev not suspiciously low
  - validate_reproducibility: Same seed + same input → identical output

Each returns: {"check_name", "passed", "severity", "message", "details"}
"""
from __future__ import annotations

import re
import math
from typing import Any, Callable


def _result(
    check_name: str,
    passed: bool,
    severity: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard validation result dict."""
    return {
        "check_name": check_name,
        "passed": passed,
        "severity": severity,
        "message": message,
        "details": details or {},
    }


# ── 1. GC Bounds ──────────────────────────────────────────────────────────────

def validate_gc_bounds(
    sequence: str,
    gc_min: float = 0.40,
    gc_max: float = 0.60,
) -> dict[str, Any]:
    """
    Validate that overall GC content is within the specified range.

    Parameters
    ----------
    sequence : str
        RNA or DNA sequence (case-insensitive, supports U/T).
    gc_min : float
        Minimum acceptable GC fraction (default 0.40).
    gc_max : float
        Maximum acceptable GC fraction (default 0.60).

    Returns
    -------
    dict with check_name, passed, severity, message, details
    """
    if not sequence:
        return _result(
            "gc_bounds", False, "critical",
            "Empty sequence provided",
            {"gc_fraction": None, "gc_min": gc_min, "gc_max": gc_max},
        )

    seq_upper = sequence.upper()
    gc_count = sum(1 for nt in seq_upper if nt in "GC")
    total = len(seq_upper)
    gc_frac = gc_count / total

    passed = gc_min <= gc_frac <= gc_max
    severity = "info" if passed else "warning"
    message = (
        f"GC content: {gc_frac:.3f} ({gc_frac*100:.1f}%) — "
        f"{'within' if passed else 'outside'} target range [{gc_min*100:.0f}%–{gc_max*100:.0f}%]"
    )
    return _result(
        "gc_bounds", passed, severity, message,
        {"gc_fraction": round(gc_frac, 4), "gc_min": gc_min, "gc_max": gc_max},
    )


# ── 2. Kozak Consensus ───────────────────────────────────────────────────────

# Kozak consensus: GCC(A|G)CCAUGG — positions -6 to +4 around AUG
# The pattern matches the 5'UTR–ORF junction
KOZAK_PATTERN = re.compile(r"GCC[AG]CCAUGG", re.IGNORECASE)


def validate_kozak_consensus(
    utr5_seq: str,
    orf_seq: str = "",
) -> dict[str, Any]:
    """
    Validate that the 5'UTR–ORF junction matches the Kozak consensus
    sequence GCC(A/G)CCAUGG.

    Parameters
    ----------
    utr5_seq : str
        5'UTR sequence (should end near AUG start).
    orf_seq : str
        ORF sequence (should start with AUG).

    Returns
    -------
    dict with check_name, passed, severity, message, details
    """
    # Build the junction region: last 12 nt of 5'UTR + first 4 nt of ORF
    # Kozak is GCC(A/G)CC + AUGG (where AUG is the start codon, 10 chars total)
    utr5_tail = utr5_seq[-12:] if len(utr5_seq) >= 12 else utr5_seq
    junction = utr5_tail + (orf_seq[:4] if orf_seq else "")
    junction_upper = junction.upper().replace("T", "U")

    match = KOZAK_PATTERN.search(junction_upper)
    passed = match is not None

    severity = "warning" if not passed else "info"
    message = (
        f"Kozak consensus {'found' if passed else 'NOT found'} at 5'UTR–ORF junction. "
        f"Junction: ...{junction_upper[-11:]}"
    )
    return _result(
        "kozak_consensus", passed, severity, message,
        {"junction_sequence": junction_upper, "pattern": "GCC[AG]CCAUGG"},
    )


# ── 3. Homopolymer & Dinucleotide Repeats ────────────────────────────────────

def validate_homopolymer_dinuc(
    sequence: str,
    max_homopolymer: int = 4,
    max_dinuc_repeat: int = 6,
) -> dict[str, Any]:
    """
    Validate that the sequence has no homopolymer runs > max_homopolymer
    and no dinucleotide repeats > max_dinuc_repeat.

    Parameters
    ----------
    sequence : str
        RNA/DNA sequence.
    max_homopolymer : int
        Maximum allowed homopolymer run length (default 4).
    max_dinuc_repeat : int
        Maximum allowed consecutive dinucleotide repeat count (default 6).

    Returns
    -------
    dict with check_name, passed, severity, message, details
    """
    if not sequence:
        return _result(
            "homopolymer_dinuc", False, "critical",
            "Empty sequence provided",
            {"homopolymer_violations": [], "dinuc_violations": []},
        )

    seq_upper = sequence.upper()
    homopolymer_violations = []
    dinuc_violations = []

    # Check homopolymer runs
    i = 0
    while i < len(seq_upper):
        run_len = 1
        while i + run_len < len(seq_upper) and seq_upper[i + run_len] == seq_upper[i]:
            run_len += 1
        if run_len > max_homopolymer:
            homopolymer_violations.append({
                "position": i,
                "nucleotide": seq_upper[i],
                "length": run_len,
            })
        i += run_len

    # Check dinucleotide repeats (e.g., ACACAC = 3 repeats of AC)
    for di_start in range(len(seq_upper) - 1):
        dinuc = seq_upper[di_start:di_start + 2]
        if len(dinuc) < 2 or dinuc[0] == dinuc[1]:
            continue  # Skip homopolymers (already checked above)
        repeat_count = 1
        pos = di_start + 2
        while pos + 1 < len(seq_upper) and seq_upper[pos:pos + 2] == dinuc:
            repeat_count += 1
            pos += 2
        if repeat_count > max_dinuc_repeat:
            # Only record if we haven't already found this exact run
            if not dinuc_violations or dinuc_violations[-1]["position"] + dinuc_violations[-1]["repeat_count"] * 2 <= di_start:
                dinuc_violations.append({
                    "position": di_start,
                    "dinucleotide": dinuc,
                    "repeat_count": repeat_count,
                })

    passed = len(homopolymer_violations) == 0 and len(dinuc_violations) == 0
    severity = "warning" if not passed else "info"

    messages = []
    if homopolymer_violations:
        messages.append(
            f"{len(homopolymer_violations)} homopolymer run(s) > {max_homopolymer} nt"
        )
    if dinuc_violations:
        messages.append(
            f"{len(dinuc_violations)} dinucleotide repeat(s) > {max_dinuc_repeat}"
        )
    message = "; ".join(messages) if messages else "No problematic repeats found"

    return _result(
        "homopolymer_dinuc", passed, severity, message,
        {
            "homopolymer_violations": homopolymer_violations[:5],
            "dinuc_violations": dinuc_violations[:5],
            "max_homopolymer": max_homopolymer,
            "max_dinuc_repeat": max_dinuc_repeat,
        },
    )


# ── 4. MFE Sanity ────────────────────────────────────────────────────────────

def validate_mfe_sanity(
    mfe: float,
    sequence_length: int,
    min_mfe_per_nt: float = -0.6,
    max_mfe_per_nt: float = -0.01,
) -> dict[str, Any]:
    """
    Validate that the MFE (minimum free energy) is negative and within
    a plausible range for the sequence length.

    Parameters
    ----------
    mfe : float
        Minimum free energy value (kcal/mol), should be negative.
    sequence_length : int
        Length of the sequence in nucleotides.
    min_mfe_per_nt : float
        Most negative allowed MFE per nucleotide (default -0.6).
    max_mfe_per_nt : float
        Least negative allowed MFE per nucleotide (default -0.01).

    Returns
    -------
    dict with check_name, passed, severity, message, details
    """
    if sequence_length <= 0:
        return _result(
            "mfe_sanity", False, "critical",
            "Invalid sequence length (≤ 0)",
            {"mfe": mfe, "sequence_length": sequence_length},
        )

    if mfe >= 0:
        return _result(
            "mfe_sanity", False, "critical",
            f"MFE is non-negative ({mfe:.2f} kcal/mol) — biologically implausible for RNA",
            {"mfe": mfe, "sequence_length": sequence_length, "mfe_per_nt": mfe / sequence_length},
        )

    mfe_per_nt = mfe / sequence_length
    passed = min_mfe_per_nt <= mfe_per_nt <= max_mfe_per_nt
    severity = "warning" if not passed else "info"

    if mfe_per_nt < min_mfe_per_nt:
        detail = "MFE too negative — may indicate overly stable structure"
    elif mfe_per_nt > max_mfe_per_nt:
        detail = "MFE not negative enough — may indicate unstructured/degraded RNA"
    else:
        detail = "MFE within expected range"

    message = (
        f"MFE: {mfe:.1f} kcal/mol ({mfe_per_nt:.4f} per nt) — {detail}"
    )
    return _result(
        "mfe_sanity", passed, severity, message,
        {
            "mfe": mfe,
            "sequence_length": sequence_length,
            "mfe_per_nt": round(mfe_per_nt, 4),
            "min_mfe_per_nt": min_mfe_per_nt,
            "max_mfe_per_nt": max_mfe_per_nt,
        },
    )


# ── 5. Score Distribution ────────────────────────────────────────────────────

def validate_score_distribution(
    scores: list[float],
    min_std_dev: float = 0.001,
    min_unique_fraction: float = 0.1,
) -> dict[str, Any]:
    """
    Validate that the score distribution is not suspiciously uniform
    (all identical or near-zero standard deviation).

    Parameters
    ----------
    scores : list[float]
        List of composite scores from the library.
    min_std_dev : float
        Minimum acceptable standard deviation (default 0.001).
    min_unique_fraction : float
        Minimum fraction of unique values (default 0.1 = 10%).

    Returns
    -------
    dict with check_name, passed, severity, message, details
    """
    if not scores:
        return _result(
            "score_distribution", False, "critical",
            "No scores provided",
            {"count": 0},
        )

    n = len(scores)
    if n == 1:
        return _result(
            "score_distribution", True, "info",
            "Only one score — distribution check not applicable",
            {"count": 1, "std_dev": 0.0, "unique_fraction": 1.0},
        )

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std_dev = math.sqrt(variance)
    unique_count = len(set(round(s, 10) for s in scores))
    unique_frac = unique_count / n

    issues = []
    if std_dev < min_std_dev:
        issues.append(f"std dev too low ({std_dev:.6f} < {min_std_dev})")
    if unique_frac < min_unique_fraction:
        issues.append(f"too few unique values ({unique_frac:.2%} < {min_unique_fraction:.0%})")

    passed = len(issues) == 0
    severity = "critical" if not passed else "info"
    message = (
        f"Score distribution: mean={mean:.4f}, std={std_dev:.4f}, "
        f"{unique_count}/{n} unique values"
        + (f" — ISSUES: {'; '.join(issues)}" if issues else " — OK")
    )
    return _result(
        "score_distribution", passed, severity, message,
        {
            "count": n,
            "mean": round(mean, 6),
            "std_dev": round(std_dev, 6),
            "unique_count": unique_count,
            "unique_fraction": round(unique_frac, 4),
            "min_std_dev": min_std_dev,
            "min_unique_fraction": min_unique_fraction,
        },
    )


# ── 6. Reproducibility ───────────────────────────────────────────────────────

def validate_reproducibility(
    pipeline_fn: Callable[..., list[dict[str, Any]]],
    *args: Any,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Validate that running the pipeline twice with the same seed produces
    identical output.

    Parameters
    ----------
    pipeline_fn : callable
        A function that takes (*args, seed=seed, **kwargs) and returns
        a list of construct dicts (each must have a 'composite_score' key).
    *args : Any
        Positional arguments to pass to pipeline_fn.
    seed : int
        Random seed to use for both runs.
    **kwargs : Any
        Keyword arguments to pass to pipeline_fn.

    Returns
    -------
    dict with check_name, passed, severity, message, details
    """
    import random as _random
    import numpy as _np

    def _run_with_seed() -> list[dict[str, Any]]:
        _random.seed(seed)
        _np.random.seed(seed)
        return pipeline_fn(*args, seed=seed, **kwargs)

    try:
        run1 = _run_with_seed()
        run2 = _run_with_seed()
    except Exception as e:
        return _result(
            "reproducibility", False, "critical",
            f"Pipeline execution failed: {e}",
            {"error": str(e)},
        )

    if len(run1) != len(run2):
        return _result(
            "reproducibility", False, "critical",
            f"Output lengths differ: run1={len(run1)}, run2={len(run2)}",
            {"run1_length": len(run1), "run2_length": len(run2)},
        )

    mismatches = 0
    for i, (c1, c2) in enumerate(zip(run1, run2)):
        score1 = c1.get("composite_score", c1.get("score"))
        score2 = c2.get("composite_score", c2.get("score"))
        if score1 != score2:
            mismatches += 1

    passed = mismatches == 0
    severity = "critical" if not passed else "info"
    message = (
        f"Reproducibility: {'PASSED' if passed else 'FAILED'} — "
        f"{mismatches}/{len(run1)} constructs differ between runs (seed={seed})"
    )
    return _result(
        "reproducibility", passed, severity, message,
        {"seed": seed, "num_constructs": len(run1), "mismatches": mismatches},
    )


# ── Batch validation helper ──────────────────────────────────────────────────

def run_all_validations(
    constructs: list[dict[str, Any]],
    gc_min: float = 0.40,
    gc_max: float = 0.60,
    max_homopolymer: int = 4,
    max_dinuc_repeat: int = 6,
    checks_enabled: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """
    Run all enabled validation checks on a library of constructs.

    Parameters
    ----------
    constructs : list of dict
        Each dict should have keys like 'full_sequence', 'utr5_seq',
        'orf_seq', 'mfe', 'composite_score'.
    gc_min, gc_max : float
        GC bounds for validate_gc_bounds.
    max_homopolymer, max_dinuc_repeat : int
        Thresholds for validate_homopolymer_dinuc.
    checks_enabled : dict
        Which checks to run (default: all enabled).

    Returns
    -------
    list of validation result dicts
    """
    enabled = checks_enabled or {
        "gc_bounds": True,
        "kozak_consensus": True,
        "homopolymer_dinuc": True,
        "mfe_sanity": True,
        "score_distribution": True,
    }

    results = []

    # Per-construct checks
    for construct in constructs:
        seq = construct.get("full_sequence", construct.get("mRNA_Sequence", ""))
        utr5 = construct.get("utr5_seq", construct.get("UTR5", ""))
        orf = construct.get("orf_seq", construct.get("ORF", ""))

        if enabled.get("gc_bounds"):
            results.append(validate_gc_bounds(seq, gc_min=gc_min, gc_max=gc_max))

        if enabled.get("kozak_consensus"):
            results.append(validate_kozak_consensus(utr5, orf))

        if enabled.get("homopolymer_dinuc"):
            results.append(validate_homopolymer_dinuc(
                seq, max_homopolymer=max_homopolymer, max_dinuc_repeat=max_dinuc_repeat
            ))

        if enabled.get("mfe_sanity"):
            mfe = construct.get("mfe", construct.get("MFE", None))
            seq_len = len(seq) if seq else 0
            if mfe is not None and seq_len > 0:
                results.append(validate_mfe_sanity(mfe, seq_len))

    # Library-level checks
    if enabled.get("score_distribution"):
        scores = [
            c.get("composite_score", c.get("Composite_Score", None))
            for c in constructs
        ]
        scores = [s for s in scores if s is not None]
        if scores:
            results.append(validate_score_distribution(scores))

    return results
