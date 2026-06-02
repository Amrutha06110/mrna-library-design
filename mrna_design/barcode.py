"""
barcode.py — Generate unique, error-correcting DNA barcodes
for each mRNA construct in the library.

Barcode modes
-------------
dna       : Plain nucleotide barcode inserted into 3'UTR
peptide   : Short peptide-encoding sequence (Rhym et al. Nat Biomed Eng 2023)
quart     : Retron-RT-compatible barcode (QuART system, 2025)
"""
from __future__ import annotations

import random
import string
from typing import Literal

BASES = list("ACGU")

BarcodeModeT = Literal["dna", "peptide", "quart"]


# ── Utility ───────────────────────────────────────────────────────────────────

def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two equal-length strings."""
    if len(a) != len(b):
        return abs(len(a) - len(b)) + sum(x != y for x, y in zip(a, b))
    return sum(x != y for x, y in zip(a, b))


def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for b in seq if b in "GC") / len(seq)


def has_homopolymer(seq: str, run_len: int = 4) -> bool:
    """Return True if seq has a homopolymer run of >= run_len."""
    for b in BASES:
        if b * run_len in seq:
            return True
    return False


def encode_peptide_barcode(seq: str) -> str:
    """
    Translate a nucleotide barcode triplet-by-triplet into single-letter
    amino acid codes (stops excluded). Returns amino acid string.
    Used for peptide-encoding barcodes (LC-MS/MS readout).
    """
    codon_table = {
        "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
        "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
        "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
        "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
        "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
        "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "UAU": "Y", "UAC": "Y", "CAU": "H", "CAC": "H",
        "CAA": "Q", "CAG": "Q", "AAU": "N", "AAC": "N",
        "AAA": "K", "AAG": "K", "GAU": "D", "GAC": "D",
        "GAA": "E", "GAG": "E", "UGU": "C", "UGC": "C",
        "UGG": "W", "CGU": "R", "CGC": "R", "CGA": "R",
        "CGG": "R", "AGA": "R", "AGG": "R", "AGU": "S",
        "AGC": "S", "GGU": "G", "GGC": "G", "GGA": "G",
        "GGG": "G",
    }
    peptide = []
    for i in range(0, len(seq) - 2, 3):
        aa = codon_table.get(seq[i:i+3])
        if aa:
            peptide.append(aa)
    return "".join(peptide)


# ── Barcode generators ────────────────────────────────────────────────────────

def _random_barcode(length: int) -> str:
    return "".join(random.choice(BASES) for _ in range(length))


def generate_barcode_pool(
    n: int,
    length: int = 16,
    min_hamming: int = 3,
    gc_min: float = 0.40,
    gc_max: float = 0.60,
    max_homopolymer: int = 4,
    max_attempts: int = 50_000,
) -> list[str]:
    """
    Generate a pool of n unique barcodes satisfying:
      - length == `length`
      - pairwise Hamming distance >= min_hamming
      - GC content in [gc_min, gc_max]
      - no homopolymer run >= max_homopolymer
    """
    pool: list[str] = []
    attempts = 0

    while len(pool) < n and attempts < max_attempts:
        attempts += 1
        bc = _random_barcode(length)

        gc = gc_fraction(bc)
        if not (gc_min <= gc <= gc_max):
            continue
        if has_homopolymer(bc, max_homopolymer):
            continue
        if any(hamming_distance(bc, existing) < min_hamming for existing in pool):
            continue

        pool.append(bc)

    if len(pool) < n:
        raise RuntimeError(
            f"Could only generate {len(pool)}/{n} valid barcodes after "
            f"{max_attempts} attempts. Try relaxing constraints "
            f"(min_hamming, gc range, or barcode length)."
        )
    return pool


# ── Public API ────────────────────────────────────────────────────────────────

def assign_barcodes(
    library: list[dict],
    bc_cfg: dict | None = None,
) -> list[dict]:
    """
    Assign a unique barcode to every construct in library.
    Adds fields: barcode, barcode_mode, barcode_peptide (if mode=peptide).
    """
    cfg = bc_cfg or {}
    mode: BarcodeModeT = cfg.get("mode", "dna")
    length      = int(cfg.get("length", 16))
    min_hamming = int(cfg.get("min_hamming", 3))
    gc_min      = float(cfg.get("gc_min", 0.40))
    gc_max      = float(cfg.get("gc_max", 0.60))

    # For QuART mode, require length divisible by 3
    if mode == "quart":
        length = (length // 3) * 3
        length = max(length, 12)

    n = len(library)
    barcodes = generate_barcode_pool(
        n=n,
        length=length,
        min_hamming=min_hamming,
        gc_min=gc_min,
        gc_max=gc_max,
    )

    for construct, bc in zip(library, barcodes):
        construct["barcode"]      = bc
        construct["barcode_mode"] = mode
        construct["barcode_gc"]   = round(gc_fraction(bc), 4)

        if mode == "peptide":
            # Ensure length is multiple of 3 for clean translation
            bc3 = bc[: (len(bc) // 3) * 3]
            construct["barcode_peptide"] = encode_peptide_barcode(bc3)

        elif mode == "quart":
            # QuART: barcode acts as retron RT template; mark position
            construct["barcode_position"] = "3utr_distal"

    return library
