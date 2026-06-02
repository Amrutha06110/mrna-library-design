"""
scorer.py — Score mRNA constructs on multiple properties
and compute a weighted composite score.

Real-world usage: swap the proxy functions below with:
  - ViennaRNA (RNAfold) for MFE
  - mRNABERT or mRNA-LM for translation efficiency
  - LinearDesign / VaxPress for CAI + structure jointly
"""
from __future__ import annotations

import math
from typing import Any

DEFAULT_WEIGHTS = {
    "cai":            0.30,
    "mfe_stability":  0.25,
    "gc_content":     0.20,
    "utr_access":     0.25,
}

# Human high-frequency codon table (relative adaptiveness ≥ 0.8)
HIGH_FREQ_CODONS = {
    "UUC", "CUG", "AUC", "GUG", "UGG", "GAG", "AAG",
    "CGC", "GGC", "CAG", "GAC", "AAC", "UGC", "ACC",
}


# ── Individual scoring functions ──────────────────────────────────────────────

def calc_gc_content(seq: str) -> float:
    """Return fraction of G/C nucleotides (0–1)."""
    if not seq:
        return 0.0
    return sum(1 for nt in seq if nt in "GC") / len(seq)


def gc_score(seq: str) -> float:
    """
    Score GC content — optimum is ~50–55% for mRNA stability.
    Returns 0–1 (1 = perfect).
    """
    gc = calc_gc_content(seq)
    deviation = abs(gc - 0.52)
    return max(0.0, 1.0 - deviation * 3.0)


def cai_score(orf_seq: str) -> float:
    """
    Proxy Codon Adaptation Index — fraction of codons
    that are high-frequency in human cells.
    Range: 0–1.

    Replace with Bio.SeqUtils.CodonAdaptationIndex for production.
    """
    codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq) - 2, 3)]
    if not codons:
        return 0.0
    high_freq = sum(1 for c in codons if c in HIGH_FREQ_CODONS)
    return high_freq / len(codons)


def mfe_stability_score(seq: str) -> float:
    """
    Proxy MFE stability — uses GC content + absence of long poly-U runs
    as a rough proxy. True MFE requires ViennaRNA / RNAfold.

    Replace with:
        import RNA
        structure, mfe = RNA.fold(seq.replace('U','T'))
    """
    gc = calc_gc_content(seq)
    # Penalise poly-U stretches (instability signal)
    poly_u_runs = sum(1 for i in range(len(seq)-4) if seq[i:i+5] == "UUUUU")
    penalty = min(poly_u_runs * 0.05, 0.3)
    # Optimal GC ~55% for structural stability
    base = 0.4 + gc * 0.6
    return max(0.0, min(1.0, base - penalty))


def utr_accessibility_score(utr5_seq: str, utr3_seq: str) -> float:
    """
    Proxy for 5' and 3' UTR ribosome/factor accessibility.
    - 5'UTR: low GC near AUG improves ribosome access (optimal ~35–45%)
    - 3'UTR: moderate GC for stability (optimal ~45–55%)

    Replace with structure prediction (RNAfold, mRNABERT embeddings).
    """
    gc5 = calc_gc_content(utr5_seq)
    gc3 = calc_gc_content(utr3_seq)
    score5 = max(0.0, 1.0 - abs(gc5 - 0.40) * 4.0)
    score3 = max(0.0, 1.0 - abs(gc3 - 0.50) * 3.0)
    return (score5 + score3) / 2.0


def uridine_depletion_score(orf_seq: str) -> float:
    """
    Higher score = fewer uridines (beneficial for innate immune evasion
    with N1-methylpseudouridine-modified mRNA).
    """
    if not orf_seq:
        return 0.0
    u_fraction = sum(1 for nt in orf_seq if nt == "U") / len(orf_seq)
    return max(0.0, 1.0 - u_fraction * 4.0)


# ── Composite scorer ──────────────────────────────────────────────────────────

def score_construct(construct: dict, weights: dict[str, float]) -> dict:
    """Add individual and composite scores to a construct dict."""
    full_seq  = construct["full_sequence"]
    orf_seq   = construct["orf_seq"]
    utr5_seq  = construct["utr5_seq"]
    utr3_seq  = construct["utr3_seq"]

    scores = {
        "cai":           cai_score(orf_seq),
        "mfe_stability": mfe_stability_score(full_seq),
        "gc_content":    calc_gc_content(full_seq),
        "gc_score":      gc_score(full_seq),
        "utr_access":    utr_accessibility_score(utr5_seq, utr3_seq),
        "u_depletion":   uridine_depletion_score(orf_seq),
    }

    # Weighted composite (only scored dimensions that have a weight)
    total_weight = sum(weights.get(k, 0) for k in ["cai", "mfe_stability", "gc_score", "utr_access"])
    if total_weight == 0:
        composite = sum(scores[k] for k in ["cai", "mfe_stability", "gc_score", "utr_access"]) / 4
    else:
        composite = (
            weights.get("cai", 0)          * scores["cai"] +
            weights.get("mfe_stability", 0) * scores["mfe_stability"] +
            weights.get("gc_content", 0)    * scores["gc_score"] +
            weights.get("utr_access", 0)    * scores["utr_access"]
        ) / total_weight

    return {**construct, **scores, "composite_score": round(composite, 6)}


def score_library(library: list[dict], weights: dict | None = None) -> list[dict]:
    """Score every construct in the library."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return [score_construct(c, w) for c in library]
