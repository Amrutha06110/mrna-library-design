"""
qc.py — Sequence quality control and validation for mRNA constructs.

Checks for:
  - CpG dinucleotide density (immunogenicity risk)
  - Homopolymer runs (synthesis failure risk)
  - Local GC extremes (folding/synthesis issues)
  - Restriction enzyme sites (cloning compatibility)
  - Cryptic splice sites and premature polyadenylation signals
  - Potential miRNA binding sites (stability risk)
  - Internal AUG codons in UTRs (leaky scanning risk)
  - dsRNA-forming inverted repeats (innate immune activation)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


# ── Common restriction sites to avoid in synthetic mRNA constructs ────────────
RESTRICTION_SITES = {
    "EcoRI":   "GAAUUC",
    "BamHI":   "GGAUCC",
    "HindIII": "AAGCUU",
    "NotI":    "GCGGCCGC",
    "XhoI":    "CUCGAG",
    "BsaI":    "GGUCUC",
    "BbsI":    "GAAGAC",
    "SapI":    "GAAGAGC",
}

# Polyadenylation signals to avoid in CDS and UTRs (premature cleavage)
POLYA_SIGNALS = [
    "AAUAAA",   # canonical
    "AUUAAA",   # variant
    "AAUACA",
    "AAUAGA",
]

# Known human miRNA seed sequences (top expressed, 7-mer seed matches)
# These represent the 2-8 nt seed region of abundant miRNAs
MIRNA_SEEDS_TOP20 = [
    "AGCAGCA",  # miR-17/20 family
    "AAAGUGC",  # miR-17 family
    "ACAGUAC",  # miR-101
    "GAGGUAG",  # let-7 family
    "UAAAGCU",  # miR-320
    "AUCACAU",  # miR-155
    "GCAGCAU",  # miR-15/16 family
    "AACACUG",  # miR-148/152 family
    "AAGGCAC",  # miR-196
    "UCCAGUU",  # miR-1/206
]


@dataclass
class QCResult:
    """Result of a single QC check."""
    check_name: str
    passed: bool
    severity: Literal["info", "warning", "critical"]
    message: str
    positions: list[int] = field(default_factory=list)


@dataclass
class QCReport:
    """Full QC report for one construct."""
    construct_id: str
    results: list[QCResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "critical")

    @property
    def warnings(self) -> list[QCResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def critical_failures(self) -> list[QCResult]:
        return [r for r in self.results if not r.passed and r.severity == "critical"]

    @property
    def score_penalty(self) -> float:
        """Compute a penalty (0–1) to subtract from composite score."""
        penalty = 0.0
        for r in self.results:
            if not r.passed:
                if r.severity == "critical":
                    penalty += 0.15
                elif r.severity == "warning":
                    penalty += 0.05
        return min(penalty, 0.5)


# ── Individual QC checks ──────────────────────────────────────────────────────

def check_cpg_density(seq: str, threshold: float = 0.02) -> QCResult:
    """
    Check CpG dinucleotide density. High CpG activates TLR9 and
    reduces expression of unmodified mRNA. Even with pseudouridine,
    minimizing CpG is best practice.

    threshold: max allowed CpG fraction (CpG count / seq length)
    """
    if not seq:
        return QCResult("cpg_density", True, "info", "Empty sequence")
    cpg_count = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == "CG")
    density = cpg_count / len(seq)
    passed = density <= threshold
    return QCResult(
        check_name="cpg_density",
        passed=passed,
        severity="warning",
        message=f"CpG density: {density:.4f} ({'OK' if passed else f'HIGH — exceeds {threshold}'}). "
                f"Found {cpg_count} CpG dinucleotides in {len(seq)} nt.",
    )


def check_uridine_content(orf_seq: str, threshold: float = 0.25) -> QCResult:
    """
    Check uridine content in ORF. High U content triggers RIG-I/MDA5
    even with N1-methylpseudouridine modification.
    """
    if not orf_seq:
        return QCResult("uridine_content", True, "info", "Empty ORF")
    u_count = sum(1 for nt in orf_seq if nt == "U")
    u_frac = u_count / len(orf_seq)
    passed = u_frac <= threshold
    return QCResult(
        check_name="uridine_content",
        passed=passed,
        severity="warning",
        message=f"Uridine fraction: {u_frac:.3f} ({'OK' if passed else f'HIGH — target < {threshold}'})",
    )


def check_homopolymer_runs(seq: str, max_run: int = 5) -> QCResult:
    """
    Flag homopolymer runs that cause polymerase slippage during
    synthesis or IVT errors.
    """
    positions = []
    i = 0
    while i < len(seq):
        run_len = 1
        while i + run_len < len(seq) and seq[i + run_len] == seq[i]:
            run_len += 1
        if run_len >= max_run:
            positions.append(i)
        i += run_len

    passed = len(positions) == 0
    return QCResult(
        check_name="homopolymer_runs",
        passed=passed,
        severity="warning",
        message=f"{'No' if passed else len(positions)} homopolymer run(s) ≥ {max_run} nt"
                + (f" at positions {positions[:5]}" if positions else ""),
        positions=positions,
    )


def check_local_gc_extremes(seq: str, window: int = 50, gc_min: float = 0.25,
                            gc_max: float = 0.80) -> QCResult:
    """
    Scan for windows with extreme GC content that cause synthesis
    failures or aberrant folding.
    """
    extreme_positions = []
    for i in range(0, len(seq) - window + 1, window // 2):
        win = seq[i:i + window]
        gc = sum(1 for nt in win if nt in "GC") / len(win)
        if gc < gc_min or gc > gc_max:
            extreme_positions.append((i, round(gc, 3)))

    passed = len(extreme_positions) == 0
    return QCResult(
        check_name="local_gc_extremes",
        passed=passed,
        severity="warning",
        message=f"{'No' if passed else len(extreme_positions)} window(s) with extreme GC"
                + (f" (first: pos {extreme_positions[0][0]}, GC={extreme_positions[0][1]})" if extreme_positions else ""),
        positions=[p[0] for p in extreme_positions],
    )


def check_restriction_sites(seq: str, sites: dict[str, str] | None = None) -> QCResult:
    """Check for unwanted restriction enzyme recognition sites."""
    sites = sites or RESTRICTION_SITES
    found = []
    for name, motif in sites.items():
        if motif in seq:
            pos = seq.index(motif)
            found.append((name, pos))

    passed = len(found) == 0
    return QCResult(
        check_name="restriction_sites",
        passed=passed,
        severity="info",
        message=f"{'No' if passed else len(found)} restriction site(s) found"
                + (f": {', '.join(n for n, _ in found[:5])}" if found else ""),
        positions=[p for _, p in found],
    )


def check_premature_polya_signals(seq: str, exclude_last: int = 50) -> QCResult:
    """
    Check for internal polyadenylation signals that could cause
    premature cleavage. Exclude the 3' end where the real signal lives.
    """
    check_seq = seq[:-exclude_last] if len(seq) > exclude_last else seq
    found_positions = []
    for signal in POLYA_SIGNALS:
        start = 0
        while True:
            pos = check_seq.find(signal, start)
            if pos == -1:
                break
            found_positions.append(pos)
            start = pos + 1

    passed = len(found_positions) == 0
    return QCResult(
        check_name="premature_polya_signal",
        passed=passed,
        severity="critical",
        message=f"{'No' if passed else len(found_positions)} internal polyA signal(s) found"
                + (f" at positions {found_positions[:5]}" if found_positions else ""),
        positions=found_positions,
    )


def check_internal_aug_in_utr(utr5_seq: str) -> QCResult:
    """
    Check for upstream AUG (uAUG) codons in the 5'UTR that could
    cause leaky scanning and reduced translation of the main ORF.
    """
    positions = [m.start() for m in re.finditer("AUG", utr5_seq)]
    passed = len(positions) == 0
    return QCResult(
        check_name="internal_aug_in_5utr",
        passed=passed,
        severity="warning",
        message=f"{'No' if passed else len(positions)} upstream AUG(s) in 5'UTR"
                + (f" at positions {positions}" if positions else "")
                + (" — may cause leaky scanning" if not passed else ""),
        positions=positions,
    )


def check_inverted_repeats(seq: str, min_stem: int = 12, max_loop: int = 6) -> QCResult:
    """
    Detect inverted repeats that could form stable hairpins / dsRNA.
    dsRNA triggers PKR, OAS, RIG-I innate immune sensors.

    Simplified scan: look for perfect inverted repeats of length ≥ min_stem.
    """
    complement = {"A": "U", "U": "A", "G": "C", "C": "G"}
    found = []

    # Scan with sliding window — only check a subset for performance
    step = max(1, len(seq) // 500)
    for i in range(0, len(seq) - min_stem * 2, step):
        stem = seq[i:i + min_stem]
        rc_stem = "".join(complement.get(nt, "N") for nt in reversed(stem))
        # Look for the reverse complement downstream within loop range
        search_start = i + min_stem
        search_end = min(i + min_stem * 2 + max_loop, len(seq))
        window = seq[search_start:search_end]
        if rc_stem in window:
            found.append(i)
            if len(found) >= 5:
                break

    passed = len(found) == 0
    return QCResult(
        check_name="inverted_repeats",
        passed=passed,
        severity="warning",
        message=f"{'No' if passed else len(found)} potential dsRNA-forming inverted repeat(s)"
                + (f" (first at pos {found[0]})" if found else ""),
        positions=found,
    )


def check_mirna_seed_matches(utr3_seq: str, seeds: list[str] | None = None) -> QCResult:
    """
    Scan 3'UTR for miRNA seed sequence matches (7-mer).
    Matches can destabilize the mRNA via RISC-mediated degradation.
    """
    seeds = seeds or MIRNA_SEEDS_TOP20
    complement = {"A": "U", "U": "A", "G": "C", "C": "G"}
    found = []

    for seed in seeds:
        # miRNA seeds bind as reverse complement
        target = "".join(complement.get(nt, "N") for nt in reversed(seed))
        if target in utr3_seq:
            found.append(seed)

    passed = len(found) <= 2  # Allow up to 2 (unavoidable in longer UTRs)
    return QCResult(
        check_name="mirna_seed_matches",
        passed=passed,
        severity="info" if passed else "warning",
        message=f"{len(found)} miRNA seed match(es) in 3'UTR"
                + (f": {', '.join(found[:3])}" if found else ""),
    )


# ── Full QC pipeline ─────────────────────────────────────────────────────────

def run_qc(construct: dict, config: dict | None = None) -> QCReport:
    """
    Run all QC checks on a construct dict.
    Returns a QCReport with all results.
    """
    cfg = config or {}
    full_seq = construct.get("full_sequence", "")
    orf_seq = construct.get("orf_seq", "")
    utr5_seq = construct.get("utr5_seq", "")
    utr3_seq = construct.get("utr3_seq", "")

    report = QCReport(construct_id=construct.get("id", "unknown"))

    report.results.append(check_cpg_density(
        full_seq, threshold=cfg.get("cpg_max_density", 0.02)))
    report.results.append(check_uridine_content(
        orf_seq, threshold=cfg.get("uridine_max_fraction", 0.25)))
    report.results.append(check_homopolymer_runs(
        full_seq, max_run=cfg.get("max_homopolymer_run", 5)))
    report.results.append(check_local_gc_extremes(
        full_seq,
        window=cfg.get("gc_window", 50),
        gc_min=cfg.get("local_gc_min", 0.25),
        gc_max=cfg.get("local_gc_max", 0.80),
    ))
    report.results.append(check_restriction_sites(full_seq))
    report.results.append(check_premature_polya_signals(full_seq))
    report.results.append(check_internal_aug_in_utr(utr5_seq))
    report.results.append(check_inverted_repeats(full_seq))
    report.results.append(check_mirna_seed_matches(utr3_seq))

    return report


def qc_library(library: list[dict], config: dict | None = None) -> list[dict]:
    """
    Run QC on every construct, adding qc_passed, qc_warnings, qc_penalty fields.
    """
    for construct in library:
        report = run_qc(construct, config)
        construct["qc_passed"] = report.passed
        construct["qc_warnings"] = len(report.warnings)
        construct["qc_critical"] = len(report.critical_failures)
        construct["qc_penalty"] = report.score_penalty
        # Adjust composite score if present
        if "composite_score" in construct:
            construct["composite_score_adjusted"] = round(
                max(0.0, construct["composite_score"] - report.score_penalty), 6
            )
    return library
