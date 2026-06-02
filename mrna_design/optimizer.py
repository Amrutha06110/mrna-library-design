"""
optimizer.py — Codon optimization for ORF sequences.

Provides:
  - Built-in multi-objective optimizer (CAI + uridine depletion + CpG avoidance)
  - Single-objective max_cai and weighted stochastic modes
  - Plug-in hooks for external tools (VaxPress, LinearDesign, CodonTransformer)
"""
from __future__ import annotations

import random
import math
from pathlib import Path


# Human codon usage table: {amino_acid: [(codon, relative_freq), ...]}
# Derived from Kazusa database (Homo sapiens)
HUMAN_CODON_TABLE: dict[str, list[tuple[str, float]]] = {
    "F": [("UUC", 0.55), ("UUU", 0.45)],
    "L": [("CUG", 0.41), ("CUC", 0.20), ("UUG", 0.13), ("CUA", 0.07), ("CUU", 0.13), ("UUA", 0.06)],
    "I": [("AUC", 0.48), ("AUU", 0.36), ("AUA", 0.16)],
    "M": [("AUG", 1.00)],
    "V": [("GUG", 0.47), ("GUC", 0.24), ("GUA", 0.11), ("GUU", 0.18)],
    "S": [("AGC", 0.24), ("UCC", 0.22), ("UCU", 0.15), ("UCA", 0.15), ("AGU", 0.15), ("UCG", 0.06)],
    "P": [("CCC", 0.33), ("CCU", 0.29), ("CCA", 0.28), ("CCG", 0.11)],
    "T": [("ACC", 0.36), ("ACA", 0.28), ("ACU", 0.24), ("ACG", 0.12)],
    "A": [("GCC", 0.40), ("GCU", 0.27), ("GCA", 0.23), ("GCG", 0.11)],
    "Y": [("UAC", 0.57), ("UAU", 0.43)],
    "H": [("CAC", 0.59), ("CAU", 0.41)],
    "Q": [("CAG", 0.75), ("CAA", 0.25)],
    "N": [("AAC", 0.54), ("AAU", 0.46)],
    "K": [("AAG", 0.58), ("AAA", 0.42)],
    "D": [("GAC", 0.54), ("GAU", 0.46)],
    "E": [("GAG", 0.58), ("GAA", 0.42)],
    "C": [("UGC", 0.55), ("UGU", 0.45)],
    "W": [("UGG", 1.00)],
    "R": [("CGC", 0.19), ("AGG", 0.22), ("AGA", 0.21), ("CGG", 0.21), ("CGA", 0.11), ("CGU", 0.08)],
    "G": [("GGC", 0.34), ("GGG", 0.25), ("GGA", 0.25), ("GGU", 0.16)],
    "*": [("UGA", 0.46), ("UAA", 0.29), ("UAG", 0.25)],
}

GENETIC_CODE: dict[str, str] = {
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


def translate(seq: str) -> str:
    """Translate RNA sequence to amino acids, stop = '*'."""
    aa = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        aa.append(GENETIC_CODE.get(codon, "X"))
    return "".join(aa)


def _codon_cai_score(codon: str) -> float:
    """Return the relative frequency of a codon among its synonyms."""
    aa = GENETIC_CODE.get(codon)
    if aa is None:
        return 0.0
    synonyms = HUMAN_CODON_TABLE.get(aa, [(codon, 1.0)])
    for syn, freq in synonyms:
        if syn == codon:
            return freq
    return 0.0


def _codon_u_content(codon: str) -> float:
    """Return the uridine fraction of a codon."""
    return sum(1 for nt in codon if nt == "U") / 3.0


def _codon_cpg_penalty(codon: str, next_codon: str | None = None) -> float:
    """
    Penalty for CpG dinucleotides within or spanning codons.
    Returns number of CpG occurrences (0, 1, or 2).
    """
    count = 0
    # Within codon
    for i in range(2):
        if codon[i:i+2] == "CG":
            count += 1
    # Spanning junction with next codon
    if next_codon and codon[2] == "C" and next_codon[0] == "G":
        count += 1
    return count


def optimize_codon_usage(seq: str, method: str = "max_cai") -> str:
    """
    Recode synonymous codons to maximise human codon usage.

    method='max_cai' : always pick the highest-frequency synonym
    method='weighted': sample proportional to frequency (more diversity)
    method='balanced': multi-objective (CAI + low U + low CpG)
    """
    codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
    optimized = []

    for idx, codon in enumerate(codons):
        aa = GENETIC_CODE.get(codon)
        if aa is None:
            optimized.append(codon)
            continue

        synonyms = HUMAN_CODON_TABLE.get(aa, [(codon, 1.0)])

        if method == "max_cai":
            best = max(synonyms, key=lambda x: x[1])[0]
        elif method == "weighted":
            syns, weights = zip(*synonyms)
            best = random.choices(syns, weights=weights, k=1)[0]
        elif method == "balanced":
            best = _select_balanced_codon(synonyms, codons, idx)
        else:
            best = codon

        optimized.append(best)

    return "".join(optimized)


def _select_balanced_codon(
    synonyms: list[tuple[str, float]],
    all_codons: list[str],
    idx: int,
) -> str:
    """
    Multi-objective codon selection balancing:
      - CAI (codon frequency) — weight 0.5
      - Uridine depletion     — weight 0.3
      - CpG avoidance         — weight 0.2

    Scores each synonym and picks the best composite.
    """
    next_codon = all_codons[idx + 1] if idx + 1 < len(all_codons) else None
    best_codon = synonyms[0][0]
    best_score = -1.0

    for codon, freq in synonyms:
        # CAI component: normalise frequency to [0, 1]
        max_freq = max(f for _, f in synonyms)
        cai_component = freq / max_freq if max_freq > 0 else 0

        # Uridine depletion: fewer U is better
        u_component = 1.0 - _codon_u_content(codon)

        # CpG avoidance: fewer CpG is better
        cpg_count = _codon_cpg_penalty(codon, next_codon)
        cpg_component = 1.0 - cpg_count * 0.5  # 0 CpG → 1.0, 2 CpG → 0.0

        # Weighted sum
        score = 0.5 * cai_component + 0.3 * u_component + 0.2 * cpg_component

        if score > best_score:
            best_score = score
            best_codon = codon

    return best_codon


def optimize_sequences(orf_dir: Path, cfg: dict) -> None:
    """
    In-place optimize all FASTA files in orf_dir.
    Writes _optimized.fasta alongside each original.

    cfg keys:
      method  : "max_cai" | "weighted" | "balanced"  (default: balanced)
      tool    : "builtin" | "vaxpress" | "lineardesign"
    """
    method = cfg.get("method", "balanced")
    tool = cfg.get("tool", "builtin")

    if tool != "builtin":
        print(f"    [optimizer] External tool '{tool}' selected — "
              f"make sure it's installed and on PATH.")
        print(f"    [optimizer] Falling back to built-in for now.")

    for fpath in sorted(orf_dir.glob("*.fa")) + sorted(orf_dir.glob("*.fasta")):
        from mrna_design.assembler import read_fasta
        records = read_fasta(fpath)
        optimized_records = []

        for rec in records:
            opt_seq = optimize_codon_usage(rec["seq"], method=method)
            # Verify amino acid sequence is preserved
            if translate(rec["seq"]) != translate(opt_seq):
                print(f"    WARNING: amino acid mismatch in {rec['name']} — "
                      f"keeping original")
                opt_seq = rec["seq"]
            optimized_records.append({
                "name": rec["name"] + "_opt",
                "seq":  opt_seq,
            })

        out_path = fpath.parent / (fpath.stem + "_optimized.fasta")
        with open(out_path, "w") as f:
            for rec in optimized_records:
                f.write(f">{rec['name']}\n{rec['seq']}\n")

        print(f"    → {out_path}")

