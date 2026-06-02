"""
scorer.py — Score mRNA constructs on multiple biophysical properties
and compute a weighted composite score.

Scoring metrics:
  - CAI: Codon Adaptation Index (geometric mean of relative adaptiveness)
  - MFE: Minimum Free Energy stability (ViennaRNA if available, else
          nearest-neighbour dinucleotide model)
  - GC content: optimality relative to 50-55% target
  - UTR accessibility: ribosome/factor accessibility based on structure
  - Codon pair bias: penalizes under-represented codon pairs
  - Uridine depletion: lower U = less innate immune activation
  - CpG depletion: lower CpG = less TLR9 activation
"""
from __future__ import annotations

import math
from typing import Any

# ── Try to import ViennaRNA for real MFE calculations ─────────────────────────
try:
    import RNA as _RNA
    HAS_VIENNARNA = True
except ImportError:
    HAS_VIENNARNA = False

DEFAULT_WEIGHTS = {
    "cai":            0.25,
    "mfe_stability":  0.20,
    "gc_content":     0.15,
    "utr_access":     0.15,
    "codon_pair":     0.10,
    "u_depletion":    0.10,
    "cpg_depletion":  0.05,
}

# ── Human codon usage: relative adaptiveness values (w_i) ─────────────────────
# From Kazusa codon usage database for Homo sapiens, normalised per amino acid
# w_i = frequency(codon) / frequency(most_frequent_synonym)
CODON_ADAPTIVENESS = {
    "UUU": 0.45, "UUC": 1.00, "UUA": 0.07, "UUG": 0.13,
    "CUU": 0.13, "CUC": 0.20, "CUA": 0.07, "CUG": 1.00,
    "AUU": 0.36, "AUC": 1.00, "AUA": 0.16, "AUG": 1.00,
    "GUU": 0.18, "GUC": 0.24, "GUA": 0.11, "GUG": 1.00,
    "UCU": 0.63, "UCC": 0.92, "UCA": 0.63, "UCG": 0.25,
    "AGU": 0.63, "AGC": 1.00,
    "CCU": 0.85, "CCC": 1.00, "CCA": 0.82, "CCG": 0.33,
    "ACU": 0.67, "ACC": 1.00, "ACA": 0.78, "ACG": 0.33,
    "GCU": 0.68, "GCC": 1.00, "GCA": 0.58, "GCG": 0.28,
    "UAU": 0.75, "UAC": 1.00, "UAA": 0.63, "UAG": 0.54,
    "CAU": 0.69, "CAC": 1.00, "CAA": 0.33, "CAG": 1.00,
    "AAU": 0.85, "AAC": 1.00, "AAA": 0.72, "AAG": 1.00,
    "GAU": 0.85, "GAC": 1.00, "GAA": 0.72, "GAG": 1.00,
    "UGU": 0.82, "UGC": 1.00, "UGA": 1.00, "UGG": 1.00,
    "CGU": 0.42, "CGC": 1.00, "CGA": 0.58, "CGG": 1.05,
    "AGA": 1.10, "AGG": 1.16,
    "GGU": 0.47, "GGC": 1.00, "GGA": 0.74, "GGG": 0.74,
}

# ── Codon pair scores (CPB): observed/expected ratio (log-scale) ──────────────
# Negative = under-represented in human genes (associated with attenuation)
# Only including highly penalised pairs; others default to 0
UNDERREPRESENTED_CODON_PAIRS = {
    # Pairs ending in UA or starting with UA are generally under-represented
    ("CUA", "CGA"): -0.5, ("GUA", "GUA"): -0.4, ("UUA", "UUA"): -0.6,
    ("AUA", "AUA"): -0.5, ("CUA", "UUA"): -0.5, ("UUA", "CGA"): -0.4,
    ("GUA", "CGA"): -0.4, ("AUA", "CGA"): -0.5, ("UUA", "GUA"): -0.4,
    # NNUpA patterns
    ("GCU", "ACG"): -0.3, ("CCU", "ACG"): -0.3, ("UCU", "ACG"): -0.3,
}

# ── Nearest-neighbour dinucleotide stacking energies (kcal/mol at 37°C) ───────
# RNA/RNA duplex parameters (Turner 2004, simplified)
# Used as MFE proxy when ViennaRNA is unavailable
DINUCLEOTIDE_ENERGY = {
    "AA": -0.93, "AU": -1.10, "AG": -2.08, "AC": -2.24,
    "UA": -1.33, "UU": -0.93, "UG": -2.08, "UC": -2.11,
    "GA": -2.35, "GU": -2.11, "GG": -3.26, "GC": -3.42,
    "CA": -2.11, "CU": -2.08, "CG": -2.36, "CC": -3.26,
}


# ── Individual scoring functions ──────────────────────────────────────────────

def calc_gc_content(seq: str) -> float:
    """Return fraction of G/C nucleotides (0–1)."""
    if not seq:
        return 0.0
    return sum(1 for nt in seq if nt in "GC") / len(seq)


def gc_score(seq: str) -> float:
    """
    Score GC content — optimum is ~50–55% for mRNA stability & translation.
    Uses a Gaussian-shaped penalty centered at 0.52.
    Returns 0–1 (1 = perfect).
    """
    gc = calc_gc_content(seq)
    # Gaussian penalty: σ = 0.12
    deviation = gc - 0.52
    return math.exp(-(deviation ** 2) / (2 * 0.12 ** 2))


def cai_score(orf_seq: str) -> float:
    """
    Codon Adaptation Index — geometric mean of relative adaptiveness
    values for each codon. This is the standard Sharp & Li (1987) method.

    Range: 0–1 (1 = all optimal codons).
    """
    codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq) - 2, 3)]
    if not codons:
        return 0.0

    log_sum = 0.0
    valid_count = 0
    for codon in codons:
        w = CODON_ADAPTIVENESS.get(codon)
        if w is not None and w > 0:
            log_sum += math.log(w)
            valid_count += 1

    if valid_count == 0:
        return 0.0
    return math.exp(log_sum / valid_count)


def mfe_stability_score(seq: str) -> float:
    """
    MFE-based stability score.

    If ViennaRNA is available: uses RNAfold for true thermodynamic MFE.
    Otherwise: uses nearest-neighbour dinucleotide energy model as a
    physics-based proxy (much better than simple GC counting).

    Returns 0–1 (1 = highly stable / very negative MFE).
    """
    if not seq:
        return 0.0

    if HAS_VIENNARNA:
        return _mfe_score_viennarna(seq)
    else:
        return _mfe_score_dinucleotide(seq)


def _mfe_score_viennarna(seq: str) -> float:
    """Compute MFE using ViennaRNA and normalise to 0–1."""
    # ViennaRNA can handle long sequences but gets slow; limit to 2500 nt
    eval_seq = seq[:2500] if len(seq) > 2500 else seq
    _, mfe = _RNA.fold(eval_seq)

    # Normalise: typical mRNA MFE ranges from -0.1 to -0.5 kcal/mol/nt
    # More negative = more stable
    mfe_per_nt = mfe / len(eval_seq)
    # Map [-0.5, 0.0] → [1.0, 0.0]
    score = min(1.0, max(0.0, -mfe_per_nt / 0.5))
    return score


def _mfe_score_dinucleotide(seq: str) -> float:
    """
    Physics-based MFE proxy using nearest-neighbour stacking energies.
    Estimates the average stacking propensity of the sequence.
    Penalises poly-U runs (instability signal for mRNA decay).
    """
    if len(seq) < 2:
        return 0.5

    # Sum dinucleotide energies
    total_energy = 0.0
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        total_energy += DINUCLEOTIDE_ENERGY.get(dinuc, -1.5)

    # Normalise per nucleotide
    energy_per_nt = total_energy / (len(seq) - 1)
    # Range is typically -1.0 to -3.5 kcal/mol
    # Map to 0–1: -1.0 → 0, -3.5 → 1
    score = (abs(energy_per_nt) - 1.0) / 2.5
    score = max(0.0, min(1.0, score))

    # Penalise poly-U stretches (signal mRNA decay via deadenylation)
    poly_u_runs = sum(1 for i in range(len(seq) - 4) if seq[i:i+5] == "UUUUU")
    penalty = min(poly_u_runs * 0.04, 0.25)

    return max(0.0, score - penalty)


def utr_accessibility_score(utr5_seq: str, utr3_seq: str) -> float:
    """
    Score UTR accessibility for ribosome binding and translation factors.

    5'UTR scoring:
      - Penalizes high GC near AUG (blocks 43S scanning)
      - Penalizes very short or very long 5'UTRs
      - Bonus for optimal length (50–100 nt)

    3'UTR scoring:
      - Moderate GC for PABP interaction
      - Penalizes extreme lengths

    If ViennaRNA available: also penalizes strong structure in 5'UTR.
    """
    # --- 5'UTR scoring ---
    gc5 = calc_gc_content(utr5_seq)
    # Optimal 5'UTR GC is 35-45% (less structured = better scanning)
    gc5_score = math.exp(-((gc5 - 0.40) ** 2) / (2 * 0.08 ** 2))

    # Length bonus: optimal 5'UTR is 50-100 nt
    len5 = len(utr5_seq)
    if 50 <= len5 <= 100:
        len5_score = 1.0
    elif 25 <= len5 < 50 or 100 < len5 <= 200:
        len5_score = 0.7
    elif len5 < 10:
        len5_score = 0.3
    else:
        len5_score = 0.5

    # If ViennaRNA available, compute 5'UTR structure penalty
    if HAS_VIENNARNA and len(utr5_seq) >= 10:
        _, mfe5 = _RNA.fold(utr5_seq)
        # Less negative MFE = less structure = better
        struct_penalty = max(0.0, min(0.3, abs(mfe5) / (len5 * 0.3)))
    else:
        # Proxy: high GC in first 30 nt near start implies structure
        first30_gc = calc_gc_content(utr5_seq[-30:]) if len(utr5_seq) >= 30 else gc5
        struct_penalty = max(0.0, (first30_gc - 0.50) * 0.5)

    score5 = max(0.0, gc5_score * 0.5 + len5_score * 0.3 + (1.0 - struct_penalty) * 0.2)

    # --- 3'UTR scoring ---
    gc3 = calc_gc_content(utr3_seq)
    gc3_score = math.exp(-((gc3 - 0.50) ** 2) / (2 * 0.10 ** 2))

    # 3'UTR length: 100-300 nt is typical for stable mRNAs
    len3 = len(utr3_seq)
    if 100 <= len3 <= 300:
        len3_score = 1.0
    elif 50 <= len3 < 100 or 300 < len3 <= 500:
        len3_score = 0.7
    else:
        len3_score = 0.5

    score3 = gc3_score * 0.6 + len3_score * 0.4

    return (score5 + score3) / 2.0


def codon_pair_bias_score(orf_seq: str) -> float:
    """
    Codon Pair Bias (CPB) score.

    Under-represented codon pairs in human genes are associated with
    translational attenuation (Coleman et al., Science 2008).
    This is used for vaccine attenuation but should be AVOIDED for
    therapeutic mRNA where maximal expression is desired.

    Returns 0–1 (1 = no under-represented pairs = good for expression).
    """
    codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq) - 2, 3)]
    if len(codons) < 2:
        return 0.5

    total_penalty = 0.0
    pair_count = 0
    for i in range(len(codons) - 1):
        pair = (codons[i], codons[i + 1])
        cpb = UNDERREPRESENTED_CODON_PAIRS.get(pair, 0.0)
        total_penalty += cpb
        pair_count += 1

    if pair_count == 0:
        return 0.5

    # Average penalty per pair; range is roughly [-0.6, 0]
    avg_penalty = total_penalty / pair_count
    # Map to score: 0 penalty = 1.0, -0.6 penalty = 0.0
    score = max(0.0, min(1.0, 1.0 + avg_penalty / 0.6))
    return score


def uridine_depletion_score(orf_seq: str) -> float:
    """
    Score uridine depletion in the ORF. Lower uridine content reduces
    innate immune activation (TLR7/8, RIG-I) even with N1-methylpseudouridine.

    Optimal range: 15-20% U (complete depletion hurts codon diversity).
    """
    if not orf_seq:
        return 0.0
    u_fraction = sum(1 for nt in orf_seq if nt == "U") / len(orf_seq)
    # Gaussian penalty centred at 0.17 (optimal U fraction for modified mRNA)
    return math.exp(-((u_fraction - 0.17) ** 2) / (2 * 0.05 ** 2))


def cpg_depletion_score(seq: str) -> float:
    """
    Score CpG dinucleotide depletion. CpG motifs activate TLR9 and
    reduce expression. Lower CpG density is better for therapeutic mRNA.

    Returns 0–1 (1 = low CpG = good).
    """
    if len(seq) < 2:
        return 0.5
    cpg_count = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == "CG")
    cpg_density = cpg_count / (len(seq) - 1)
    # Human genomic average CpG density is ~0.01; synthetic mRNA target < 0.005
    # Score: 0 CpG → 1.0, 0.03 → 0.0
    score = max(0.0, 1.0 - cpg_density / 0.03)
    return score


# ── Composite scorer ──────────────────────────────────────────────────────────

def score_construct(construct: dict, weights: dict[str, float]) -> dict:
    """Add individual and composite scores to a construct dict."""
    full_seq = construct["full_sequence"]
    orf_seq = construct["orf_seq"]
    utr5_seq = construct["utr5_seq"]
    utr3_seq = construct["utr3_seq"]

    scores = {
        "cai":           cai_score(orf_seq),
        "mfe_stability": mfe_stability_score(full_seq),
        "gc_content":    calc_gc_content(full_seq),
        "gc_score":      gc_score(full_seq),
        "utr_access":    utr_accessibility_score(utr5_seq, utr3_seq),
        "codon_pair":    codon_pair_bias_score(orf_seq),
        "u_depletion":   uridine_depletion_score(orf_seq),
        "cpg_depletion": cpg_depletion_score(full_seq),
    }

    # Weighted composite — use all dimensions that have weights
    scored_dims = ["cai", "mfe_stability", "gc_score", "utr_access",
                   "codon_pair", "u_depletion", "cpg_depletion"]
    total_weight = sum(weights.get(k, 0) for k in scored_dims)

    if total_weight == 0:
        composite = sum(scores.get(k, 0) for k in scored_dims) / len(scored_dims)
    else:
        composite = sum(
            weights.get(k, 0) * scores.get(k, 0) for k in scored_dims
        ) / total_weight

    return {**construct, **scores, "composite_score": round(composite, 6)}


def score_library(library: list[dict], weights: dict | None = None) -> list[dict]:
    """Score every construct in the library."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return [score_construct(c, w) for c in library]
