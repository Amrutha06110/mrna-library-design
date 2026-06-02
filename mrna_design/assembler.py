"""
assembler.py — Read FASTA files for 5'UTR, ORF, and 3'UTR,
then generate every combination as a full mRNA construct.
"""
from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Iterator


KOZAK_SEQUENCES = {
    "strong":   "GCCACCAUGG",
    "moderate": "ACCAUGG",
    "none":     "",
}

CAP_LABELS = {
    "m7G":   "m7G",
    "cap1":  "Cap-1",
    "arca":  "ARCA",
    "none":  "",
}


def read_fasta(path: Path) -> list[dict]:
    """Parse a single FASTA file into a list of {name, seq} dicts."""
    records = []
    current_name, current_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records.append({"name": current_name, "seq": "".join(current_seq)})
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper().replace("T", "U"))
    if current_name is not None:
        records.append({"name": current_name, "seq": "".join(current_seq)})
    return records


def load_fasta_dir(directory: Path) -> list[dict]:
    """Load all .fa / .fasta files from a directory."""
    records = []
    for ext in ("*.fa", "*.fasta", "*.txt"):
        for fpath in sorted(directory.glob(ext)):
            records.extend(read_fasta(fpath))
    return records


def assemble_construct(
    utr5: dict,
    orf: dict,
    utr3: dict,
    cap: str = "m7G",
    kozak: str = "strong",
    polya_len: int = 100,
) -> dict:
    """Build one full mRNA sequence from components."""
    kozak_seq = KOZAK_SEQUENCES.get(kozak, "")
    polya_tail = "A" * polya_len

    # Insert Kozak between 5'UTR and ORF if not already present
    orf_seq = orf["seq"]
    if kozak_seq and not utr5["seq"].endswith(kozak_seq) and not orf_seq.startswith("AUG"):
        junction = kozak_seq + orf_seq
    else:
        junction = orf_seq

    full_seq = utr5["seq"] + junction + utr3["seq"] + polya_tail

    return {
        "utr5_name":    utr5["name"],
        "orf_name":     orf["name"],
        "utr3_name":    utr3["name"],
        "utr5_seq":     utr5["seq"],
        "orf_seq":      orf["seq"],
        "utr3_seq":     utr3["seq"],
        "cap":          CAP_LABELS.get(cap, cap),
        "kozak":        kozak,
        "polya_length": polya_len,
        "full_sequence": full_seq,
        "total_length":  len(full_seq),
    }


def assemble_library(
    utr5_dir: Path,
    orf_dir: Path,
    utr3_dir: Path,
    cap: str = "m7G",
    kozak: str = "strong",
    polya_len: int = 100,
    max_combinations: int | None = 1000,
) -> list[dict]:
    """
    Produce all combinations of 5'UTR × ORF × 3'UTR,
    optionally capped at max_combinations (random sampling).
    """
    utr5_seqs = load_fasta_dir(utr5_dir)
    orf_seqs  = load_fasta_dir(orf_dir)
    utr3_seqs = load_fasta_dir(utr3_dir)

    if not utr5_seqs:
        raise ValueError(f"No FASTA files found in {utr5_dir}")
    if not orf_seqs:
        raise ValueError(f"No FASTA files found in {orf_dir}")
    if not utr3_seqs:
        raise ValueError(f"No FASTA files found in {utr3_dir}")

    all_combos = list(itertools.product(utr5_seqs, orf_seqs, utr3_seqs))

    if max_combinations and len(all_combos) > max_combinations:
        all_combos = random.sample(all_combos, max_combinations)

    library = []
    for i, (u5, orf, u3) in enumerate(all_combos, start=1):
        construct = assemble_construct(u5, orf, u3, cap, kozak, polya_len)
        construct["id"] = f"mRNA-{i:04d}"
        library.append(construct)

    return library
