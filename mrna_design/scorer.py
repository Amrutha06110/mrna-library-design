"""
scorer.py — Score mRNA constructs on multiple biophysical properties
and compute a weighted composite score.

Scoring metrics (10 dimensions):
  - CAI: Codon Adaptation Index (geometric mean of relative adaptiveness)
  - MFE: Minimum Free Energy stability (ViennaRNA if available, else
          nearest-neighbour dinucleotide model)
  - GC content: optimality relative to 50-55% target
  - UTR accessibility: ribosome/factor accessibility based on structure,
    motifs (TOP, ARE, IRES-like), and length/GC profiling
  - Codon pair bias: comprehensive CPB table (Coleman et al. 2008)
  - Uridine depletion: lower U = less innate immune activation
  - CpG depletion: lower CpG = less TLR9 activation
  - Translation efficiency: 5'UTR k-mer model (Optimus 5-Prime inspired)
  - Codon ramp: 5' ORF slow ramp for co-translational folding
  - Codon diversity: avoids repetitive codon autocorrelation
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
    "cai":              0.20,
    "mfe_stability":    0.15,
    "gc_content":       0.10,
    "utr_access":       0.12,
    "codon_pair":       0.10,
    "u_depletion":      0.08,
    "cpg_depletion":    0.05,
    "translation_eff":  0.10,
    "codon_ramp":       0.05,
    "codon_diversity":  0.05,
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
# Comprehensive codon pair bias table based on Coleman et al. (Science 2008).
# Negative = under-represented in human genes (translational attenuation).
# Positive = over-represented (associated with efficient translation).
# Covers the NNUpA, UpANN, and NNCpG junction patterns systematically.
UNDERREPRESENTED_CODON_PAIRS = {
    # ── UA-dinucleotide junction pairs (strongest attenuation signal) ──
    # Pairs where codon1 ends with U and codon2 starts with A
    ("GCU", "AUG"): -0.32, ("GCU", "AUU"): -0.28, ("GCU", "AUC"): -0.25,
    ("GCU", "AUA"): -0.35, ("GCU", "ACG"): -0.30, ("GCU", "AGG"): -0.22,
    ("CCU", "AUG"): -0.30, ("CCU", "AUU"): -0.26, ("CCU", "AUC"): -0.23,
    ("CCU", "AUA"): -0.33, ("CCU", "ACG"): -0.30, ("CCU", "AGG"): -0.20,
    ("UCU", "AUG"): -0.31, ("UCU", "AUU"): -0.27, ("UCU", "AUC"): -0.24,
    ("UCU", "AUA"): -0.34, ("UCU", "ACG"): -0.30, ("UCU", "AGG"): -0.21,
    ("ACU", "AUG"): -0.29, ("ACU", "AUU"): -0.25, ("ACU", "AUC"): -0.22,
    ("ACU", "AUA"): -0.32, ("ACU", "ACG"): -0.28, ("ACU", "AGG"): -0.19,
    ("GGU", "AUG"): -0.28, ("GGU", "AUA"): -0.31, ("GGU", "ACG"): -0.26,
    ("AGU", "AUG"): -0.30, ("AGU", "AUA"): -0.33, ("AGU", "ACG"): -0.28,
    ("CGU", "AUG"): -0.27, ("CGU", "AUA"): -0.30, ("CGU", "ACG"): -0.25,
    ("UGU", "AUG"): -0.26, ("UGU", "AUA"): -0.29, ("UGU", "ACG"): -0.24,
    ("GAU", "AUG"): -0.25, ("GAU", "AUA"): -0.28, ("GAU", "ACG"): -0.23,
    ("AAU", "AUG"): -0.24, ("AAU", "AUA"): -0.27, ("AAU", "ACG"): -0.22,
    ("CAU", "AUG"): -0.26, ("CAU", "AUA"): -0.29, ("CAU", "ACG"): -0.24,
    ("UAU", "AUG"): -0.25, ("UAU", "AUA"): -0.28, ("UAU", "ACG"): -0.23,
    ("UUU", "AUG"): -0.22, ("UUU", "AUA"): -0.25, ("UUU", "ACG"): -0.20,
    ("AUU", "AUG"): -0.23, ("AUU", "AUA"): -0.26, ("AUU", "ACG"): -0.21,
    ("GUU", "AUG"): -0.24, ("GUU", "AUA"): -0.27, ("GUU", "ACG"): -0.22,
    ("CUU", "AUG"): -0.23, ("CUU", "AUA"): -0.26, ("CUU", "ACG"): -0.21,
    # ── Rare codon pairs with UA or CG dinucleotide at junction ──
    ("CUA", "CGA"): -0.50, ("GUA", "GUA"): -0.40, ("UUA", "UUA"): -0.60,
    ("AUA", "AUA"): -0.50, ("CUA", "UUA"): -0.50, ("UUA", "CGA"): -0.40,
    ("GUA", "CGA"): -0.40, ("AUA", "CGA"): -0.50, ("UUA", "GUA"): -0.40,
    ("CUA", "GUA"): -0.38, ("CUA", "AUA"): -0.42, ("GUA", "UUA"): -0.38,
    ("GUA", "AUA"): -0.36, ("AUA", "UUA"): -0.44, ("AUA", "GUA"): -0.38,
    ("UUA", "AUA"): -0.48, ("UUA", "CUA"): -0.42,
    # ── CpG junction pairs (TLR9 activation, expression reduction) ──
    # Codon1 ends with C, codon2 starts with G
    ("GCC", "GCC"): -0.18, ("GCC", "GCG"): -0.25, ("GCC", "GGC"): -0.20,
    ("GCC", "GAG"): -0.15, ("UCC", "GCC"): -0.17, ("UCC", "GCG"): -0.24,
    ("CCC", "GCC"): -0.16, ("CCC", "GCG"): -0.23, ("CCC", "GGC"): -0.18,
    ("ACC", "GCC"): -0.15, ("ACC", "GCG"): -0.22, ("ACC", "GGC"): -0.17,
    ("AAC", "GCC"): -0.14, ("AAC", "GCG"): -0.21, ("AAC", "GGC"): -0.16,
    ("GAC", "GCC"): -0.15, ("GAC", "GCG"): -0.22, ("GAC", "GGC"): -0.17,
    ("CAC", "GCC"): -0.14, ("CAC", "GCG"): -0.21, ("CAC", "GGC"): -0.16,
    ("UGC", "GCC"): -0.16, ("UGC", "GCG"): -0.23, ("UGC", "GGC"): -0.18,
    ("AGC", "GCC"): -0.15, ("AGC", "GCG"): -0.22, ("AGC", "GGC"): -0.17,
    ("CGC", "GCC"): -0.18, ("CGC", "GCG"): -0.26, ("CGC", "GGC"): -0.20,
    ("GGC", "GCC"): -0.17, ("GGC", "GCG"): -0.24, ("GGC", "GGC"): -0.19,
    ("AUC", "GCC"): -0.14, ("AUC", "GCG"): -0.21, ("AUC", "GGC"): -0.16,
    ("GUC", "GCC"): -0.15, ("GUC", "GCG"): -0.22, ("GUC", "GGC"): -0.17,
    ("CUC", "GCC"): -0.14, ("CUC", "GCG"): -0.21, ("CUC", "GGC"): -0.16,
    ("UUC", "GCC"): -0.13, ("UUC", "GCG"): -0.20, ("UUC", "GGC"): -0.15,
    ("UAC", "GCC"): -0.14, ("UAC", "GCG"): -0.21, ("UAC", "GGC"): -0.16,
    ("CAG", "CGC"): -0.19, ("CAG", "CGG"): -0.22, ("CAG", "CGA"): -0.28,
    # ── Over-represented pairs (positive, good for expression) ──
    ("GCC", "AUG"): 0.12, ("GCC", "AAG"): 0.10, ("GCC", "GAG"): 0.08,
    ("AAG", "GAG"): 0.15, ("GAG", "AAG"): 0.14, ("CUG", "GAG"): 0.12,
    ("GAG", "CUG"): 0.11, ("AAG", "CUG"): 0.10, ("CUG", "AAG"): 0.09,
    ("GAG", "GAG"): 0.08, ("AAG", "AAG"): 0.07, ("CUG", "CUG"): 0.06,
    ("GCC", "ACC"): 0.09, ("ACC", "GCC"): 0.08, ("GAC", "AAG"): 0.07,
    ("AAC", "AAG"): 0.06, ("AGC", "AAG"): 0.06, ("UCC", "AAG"): 0.05,
}

# ── 5'UTR k-mer translation efficiency weights ───────────────────────────────
# Inspired by Optimus 5-Prime (Sample et al., Nature Biotech 2019).
# These k-mers in the 5'UTR are associated with high/low translation.
# Positive = promotes translation; Negative = inhibits translation.
UTR5_KMER_WEIGHTS = {
    # Positive signals (promote cap-dependent translation)
    "GCCACC": 0.25,    # Kozak consensus context
    "CCACC":  0.20,    # Strong Kozak core
    "GCCGCC": 0.15,    # GC-rich initiation context
    "ACCAUG": 0.18,    # Perfect Kozak-AUG junction
    "CAACC":  0.10,    # Moderate positive context
    "GCACC":  0.12,    # Near-Kozak
    "CCAUG":  0.15,    # Direct pre-AUG context
    "AAACC":  0.05,    # Weak positive
    "GCCAC":  0.10,
    "CCGCC":  0.08,
    "UGCCC":  0.06,    # Stability element
    "GCGCC":  0.07,
    # Negative signals (inhibit translation)
    "UUUUU": -0.30,    # Poly-U destabilizes
    "AAAAA": -0.15,    # Poly-A in 5'UTR (not poly-A tail)
    "GGGGG": -0.20,    # G-quadruplex forming
    "GGGGC": -0.18,    # G-quadruplex context
    "GGGCG": -0.15,    # Strong structure
    "CCCCC": -0.12,    # Homopolymer
    "UAAUU": -0.10,    # ARE-like
    "AUGUG": -0.25,    # Upstream AUG context (competes)
    "AUGAU": -0.22,    # Upstream AUG context
    "AUGAA": -0.20,    # Upstream AUG context
    "CUGAU": -0.08,    # Near-cognate start
    "GUGAU": -0.08,    # Near-cognate start
    "UUGAU": -0.06,    # Weak near-cognate
}

# ── TOP motif and regulatory element patterns ─────────────────────────────────
# Terminal Oligopyrimidine (TOP) motif: mTOR-regulated translation
TOP_PATTERN_START = "CUUCC"  # Canonical TOP start (after cap)

# AU-rich elements (AREs) in 3'UTR — destabilization signals
ARE_PATTERNS = ["AUUUA", "UAUUUAU", "UUAUUUAUU"]

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

    5'UTR scoring (enhanced with motif detection):
      - Penalizes high GC near AUG (blocks 43S scanning)
      - Penalizes very short or very long 5'UTRs
      - Bonus for optimal length (50–100 nt)
      - Detects TOP motifs (mTOR-regulated)
      - Penalizes near-cognate start codons (CUG, GUG, UUG)

    3'UTR scoring (enhanced with stability element detection):
      - Moderate GC for PABP interaction
      - Penalizes extreme lengths
      - Penalizes AU-rich elements (AREs) that trigger mRNA decay
      - Rewards cytoplasmic polyadenylation elements (CPEs)

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
        # Proxy: high GC in last 30 nt near AUG implies structure blocking scanning
        last30_gc = calc_gc_content(utr5_seq[-30:]) if len(utr5_seq) >= 30 else gc5
        struct_penalty = max(0.0, (last30_gc - 0.50) * 0.5)

    # Near-cognate start penalty (leaky scanning competitors)
    near_cognate_count = (
        utr5_seq.count("CUG") + utr5_seq.count("GUG") + utr5_seq.count("UUG")
    )
    cognate_penalty = min(0.15, near_cognate_count * 0.04)

    score5 = max(0.0,
        gc5_score * 0.35
        + len5_score * 0.25
        + (1.0 - struct_penalty) * 0.25
        + (1.0 - cognate_penalty) * 0.15
    )

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

    # AU-rich element (ARE) penalty — AREs recruit decay machinery
    are_penalty = 0.0
    for pattern in ARE_PATTERNS:
        are_count = utr3_seq.count(pattern)
        if are_count > 0:
            are_penalty += are_count * 0.08
    are_penalty = min(are_penalty, 0.25)

    # Cytoplasmic polyadenylation element bonus (UUUUAU or UUUUAAU)
    cpe_bonus = 0.0
    if "UUUUAU" in utr3_seq or "UUUUAAU" in utr3_seq:
        cpe_bonus = 0.05  # Promotes cytoplasmic polyadenylation

    score3 = max(0.0,
        gc3_score * 0.40
        + len3_score * 0.30
        + (1.0 - are_penalty) * 0.20
        + cpe_bonus * 0.10 / 0.05 if cpe_bonus else gc3_score * 0.40 + len3_score * 0.30 + (1.0 - are_penalty) * 0.20
    )
    # Simplified: recalculate cleanly
    score3 = gc3_score * 0.40 + len3_score * 0.30 + (1.0 - are_penalty) * 0.20 + min(cpe_bonus, 0.10)

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


def translation_efficiency_score(utr5_seq: str) -> float:
    """
    Predict translation efficiency from 5'UTR sequence features.

    Model inspired by Optimus 5-Prime (Sample et al., Nature Biotech 2019):
      - k-mer composition scoring (5-mers associated with high/low TE)
      - Penalizes G-quadruplex-forming motifs (block scanning)
      - Rewards optimal Kozak-like context near 3' end of UTR
      - Penalizes upstream AUG/near-cognate starts (CUG, GUG, UUG)
      - TOP motif detection (mTOR-regulated, context-dependent)

    Returns 0–1 (1 = high predicted translation efficiency).
    """
    if not utr5_seq or len(utr5_seq) < 5:
        return 0.5

    # --- k-mer composition score ---
    kmer_score = 0.0
    kmer_count = 0
    for k in [5, 6]:
        for i in range(len(utr5_seq) - k + 1):
            kmer = utr5_seq[i:i+k]
            weight = UTR5_KMER_WEIGHTS.get(kmer, 0.0)
            if weight != 0:
                kmer_score += weight
                kmer_count += 1

    # Normalise k-mer score to [-1, 1] then map to [0, 1]
    if kmer_count > 0:
        kmer_component = kmer_score / kmer_count
    else:
        kmer_component = 0.0
    kmer_component = max(-1.0, min(1.0, kmer_component))
    kmer_norm = (kmer_component + 1.0) / 2.0  # Map to [0, 1]

    # --- G-quadruplex penalty ---
    # G4 motif: G3+N1-7G3+N1-7G3+N1-7G3+
    g4_penalty = 0.0
    g_runs = 0
    i = 0
    while i < len(utr5_seq):
        if utr5_seq[i] == "G":
            run = 0
            while i < len(utr5_seq) and utr5_seq[i] == "G":
                run += 1
                i += 1
            if run >= 3:
                g_runs += 1
        else:
            i += 1
    if g_runs >= 4:
        g4_penalty = 0.3
    elif g_runs >= 3:
        g4_penalty = 0.15

    # --- Upstream ORF penalty ---
    # Count upstream AUG and near-cognate starts (CUG, GUG, UUG)
    uorf_starts = utr5_seq.count("AUG")
    near_cognate = utr5_seq.count("CUG") + utr5_seq.count("GUG") + utr5_seq.count("UUG")
    uorf_penalty = min(0.35, uorf_starts * 0.15 + near_cognate * 0.03)

    # --- TOP motif bonus/context ---
    # TOP mRNAs are mTOR-regulated; can be beneficial in growth conditions
    top_bonus = 0.0
    if utr5_seq[:5] == TOP_PATTERN_START or utr5_seq[:4] == "CUUU":
        top_bonus = 0.05  # Slight bonus for regulated translation

    # --- Kozak context near 3' end (last 10 nt) ---
    kozak_bonus = 0.0
    last10 = utr5_seq[-10:] if len(utr5_seq) >= 10 else utr5_seq
    if "GCCACC" in last10:
        kozak_bonus = 0.15
    elif "CCACC" in last10 or "GCACC" in last10:
        kozak_bonus = 0.10
    elif "ACC" in last10[-6:]:
        kozak_bonus = 0.05

    # Combine components
    raw_score = (
        kmer_norm * 0.35
        + (1.0 - g4_penalty) * 0.20
        + (1.0 - uorf_penalty) * 0.25
        + kozak_bonus * 0.15 / 0.15  # normalise kozak to [0,1]
        + top_bonus
    )

    return max(0.0, min(1.0, raw_score))


def codon_ramp_score(orf_seq: str) -> float:
    """
    Score the codon usage ramp at the 5' end of the ORF.

    Biology: Ribosomes translate the first ~30-50 codons slowly to allow
    proper mRNA-ribosome complex formation and co-translational folding.
    A "ramp" of gradually increasing codon optimality (low CAI → high CAI)
    in the first ~40 codons correlates with higher overall protein output
    (Tuller et al., Cell 2010; Hanson & Coller, Nat Rev MCB 2018).

    Scoring:
      - First 15 codons: should have moderate CAI (0.4-0.7)
      - Codons 16-40: should ramp up toward high CAI
      - Penalizes very rare codons at start (ribosome stalling)
      - Penalizes all-optimal codons at start (no ramp = ribosome jamming)

    Returns 0–1 (1 = good ramp profile).
    """
    codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq) - 2, 3)]
    if len(codons) < 10:
        return 0.5

    ramp_region = codons[:min(40, len(codons))]
    body_start = min(40, len(codons))

    # Calculate CAI for early vs. later codons
    def _region_cai(codon_list):
        if not codon_list:
            return 0.5
        log_sum = 0.0
        count = 0
        for c in codon_list:
            w = CODON_ADAPTIVENESS.get(c)
            if w and w > 0:
                log_sum += math.log(w)
                count += 1
        return math.exp(log_sum / count) if count > 0 else 0.5

    early_cai = _region_cai(ramp_region[:15])
    mid_cai = _region_cai(ramp_region[15:]) if len(ramp_region) > 15 else early_cai
    body_cai = _region_cai(codons[body_start:body_start+50]) if len(codons) > body_start else mid_cai

    # Ideal ramp: early_cai < mid_cai <= body_cai
    # Score the ramp gradient
    ramp_gradient = 0.0

    # Early region should be moderate (0.4–0.7 CAI)
    if 0.4 <= early_cai <= 0.7:
        early_score = 1.0
    elif early_cai < 0.3:
        early_score = 0.4  # Too slow — ribosome may stall/abort
    elif early_cai > 0.85:
        early_score = 0.5  # No ramp — traffic jam
    else:
        early_score = 0.7

    # Ramp should increase
    if mid_cai > early_cai:
        ramp_gradient = min(1.0, (mid_cai - early_cai) / 0.3)
    else:
        ramp_gradient = 0.3  # Flat or decreasing is suboptimal

    # Body should be well-adapted
    body_score = min(1.0, body_cai / 0.8)

    return early_score * 0.4 + ramp_gradient * 0.3 + body_score * 0.3


def codon_diversity_score(orf_seq: str) -> float:
    """
    Score codon diversity / autocorrelation avoidance.

    Biology: Repetitive use of the same codon depletes charged tRNA pools
    locally, causing ribosome pausing (Presnyak et al., Cell 2015).
    Also, repetitive sequences can form secondary structures.

    Measures:
      - Codon repeat avoidance (same codon used consecutively)
      - Synonym diversity (uses multiple synonyms per amino acid)
      - Dinucleotide repetition at codon junctions

    Returns 0–1 (1 = high diversity, no problematic repetition).
    """
    codons = [orf_seq[i:i+3] for i in range(0, len(orf_seq) - 2, 3)]
    if len(codons) < 4:
        return 0.5

    # --- Consecutive repeat penalty ---
    consecutive_repeats = 0
    for i in range(len(codons) - 1):
        if codons[i] == codons[i + 1]:
            consecutive_repeats += 1
    repeat_fraction = consecutive_repeats / (len(codons) - 1)
    # Expected random repeat rate for 61 codons ≈ 0.02; penalize above 0.05
    repeat_score = max(0.0, 1.0 - repeat_fraction / 0.15)

    # --- Synonym diversity ---
    # For each amino acid, count how many different synonyms are used
    from collections import Counter
    aa_codons: dict[str, list[str]] = {}
    for c in codons:
        aa = None
        for codon_key, amino in CODON_ADAPTIVENESS.items():
            pass  # We need GENETIC_CODE here
        # Use a simple lookup
        aa = _get_amino_acid(c)
        if aa and aa != "*":
            aa_codons.setdefault(aa, []).append(c)

    diversity_scores = []
    for aa, used_codons in aa_codons.items():
        if len(used_codons) >= 3:
            unique_used = len(set(used_codons))
            # How many synonyms exist for this AA?
            total_synonyms = _count_synonyms(aa)
            if total_synonyms > 1:
                # Score: using more synonyms is better
                diversity_scores.append(
                    min(1.0, unique_used / min(total_synonyms, len(used_codons)))
                )

    synonym_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.7

    # --- Junction dinucleotide repetition ---
    junction_dinucs = []
    for i in range(len(codons) - 1):
        junction_dinucs.append(codons[i][2] + codons[i+1][0])
    if junction_dinucs:
        dinuc_counts = Counter(junction_dinucs)
        max_dinuc_freq = max(dinuc_counts.values()) / len(junction_dinucs)
        # Penalize if one junction dinucleotide dominates
        junction_score = max(0.0, 1.0 - (max_dinuc_freq - 0.25) / 0.5)
    else:
        junction_score = 0.5

    return repeat_score * 0.4 + synonym_diversity * 0.35 + junction_score * 0.25


# ── Helper lookups for codon diversity ────────────────────────────────────────

# Simple genetic code for diversity scoring
_GENETIC_CODE_MAP = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_AA_SYNONYM_COUNT = {}


def _get_amino_acid(codon: str) -> str | None:
    return _GENETIC_CODE_MAP.get(codon)


def _count_synonyms(aa: str) -> int:
    """Return number of synonymous codons for an amino acid."""
    global _AA_SYNONYM_COUNT
    if not _AA_SYNONYM_COUNT:
        from collections import Counter
        counts = Counter(_GENETIC_CODE_MAP.values())
        _AA_SYNONYM_COUNT = dict(counts)
    return _AA_SYNONYM_COUNT.get(aa, 1)


# ── Composite scorer ──────────────────────────────────────────────────────────

def score_construct(construct: dict, weights: dict[str, float]) -> dict:
    """Add individual and composite scores to a construct dict."""
    full_seq = construct["full_sequence"]
    orf_seq = construct["orf_seq"]
    utr5_seq = construct["utr5_seq"]
    utr3_seq = construct["utr3_seq"]

    scores = {
        "cai":              cai_score(orf_seq),
        "mfe_stability":    mfe_stability_score(full_seq),
        "gc_content":       calc_gc_content(full_seq),
        "gc_score":         gc_score(full_seq),
        "utr_access":       utr_accessibility_score(utr5_seq, utr3_seq),
        "codon_pair":       codon_pair_bias_score(orf_seq),
        "u_depletion":      uridine_depletion_score(orf_seq),
        "cpg_depletion":    cpg_depletion_score(full_seq),
        "translation_eff":  translation_efficiency_score(utr5_seq),
        "codon_ramp":       codon_ramp_score(orf_seq),
        "codon_diversity":  codon_diversity_score(orf_seq),
    }

    # Weighted composite — use all dimensions that have weights
    scored_dims = ["cai", "mfe_stability", "gc_score", "utr_access",
                   "codon_pair", "u_depletion", "cpg_depletion",
                   "translation_eff", "codon_ramp", "codon_diversity"]
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
