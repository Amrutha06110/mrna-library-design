"""
pipeline.py — Chunked/streaming pipeline engine for mRNA Library Design.

Supports:
  - Streaming assembly to avoid holding all combinations in memory
  - Chunked scoring with optional parallel workers
  - Structured logging with stage timings
  - Scoring transparency (per-metric raw, normalized, weighted contribution)
  - Explain-top-N artifact generation
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from mrna_design.assembler import (
    assemble_construct,
    load_fasta_dir,
)
from mrna_design.config_model import PipelineConfig
from mrna_design.scorer import score_construct, DEFAULT_WEIGHTS

logger = logging.getLogger("mrna_design")


def _iter_combinations(
    utr5_seqs: list[dict],
    orf_seqs: list[dict],
    utr3_seqs: list[dict],
    max_combinations: int | None,
) -> Iterator[tuple[dict, dict, dict]]:
    """Yield (utr5, orf, utr3) tuples, optionally sampled."""
    total = len(utr5_seqs) * len(orf_seqs) * len(utr3_seqs)

    if max_combinations and total > max_combinations:
        # Sample without building full list using reservoir sampling approach
        indices = random.sample(range(total), max_combinations)
        indices.sort()
        combo_iter = itertools.product(utr5_seqs, orf_seqs, utr3_seqs)
        idx_set = set(indices)
        for i, combo in enumerate(combo_iter):
            if i in idx_set:
                yield combo
    else:
        yield from itertools.product(utr5_seqs, orf_seqs, utr3_seqs)


def _score_chunk(
    chunk: list[dict],
    weights: dict[str, float],
) -> list[dict]:
    """Score a chunk of constructs (suitable for multiprocessing)."""
    return [score_construct(c, weights) for c in chunk]


def run_pipeline(
    cfg: PipelineConfig,
    optimize: bool = False,
    no_barcode: bool = False,
    explain_top: int | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """
    Execute the full mRNA library design pipeline.

    Args:
        cfg: Validated pipeline configuration.
        optimize: Whether to run codon optimization first.
        no_barcode: Skip barcode assignment.
        explain_top: If set, generate explanation artifact for top N constructs.
        dry_run: Validate inputs and print stats without executing.

    Returns:
        Ranked library list (empty list if dry_run).
    """
    from mrna_design.barcode import assign_barcodes
    from mrna_design.optimizer import optimize_sequences
    from mrna_design.qc import qc_library

    timings: dict[str, float] = {}

    utr5_dir = Path(cfg.utr5_dir)
    orf_dir = Path(cfg.orf_dir)
    utr3_dir = Path(cfg.utr3_dir)
    output = Path(cfg.output_dir)

    # Load sequences
    t0 = time.time()
    utr5_seqs = load_fasta_dir(utr5_dir)
    orf_seqs = load_fasta_dir(orf_dir)
    utr3_seqs = load_fasta_dir(utr3_dir)
    timings["load_inputs"] = time.time() - t0

    if not utr5_seqs:
        raise ValueError(f"No FASTA files found in {utr5_dir}")
    if not orf_seqs:
        raise ValueError(f"No FASTA files found in {orf_dir}")
    if not utr3_seqs:
        raise ValueError(f"No FASTA files found in {utr3_dir}")

    total_possible = len(utr5_seqs) * len(orf_seqs) * len(utr3_seqs)
    actual_count = min(total_possible, cfg.max_combinations) if cfg.max_combinations else total_possible

    logger.info(
        "Input stats: %d UTR5 × %d ORF × %d UTR3 = %d possible combinations",
        len(utr5_seqs), len(orf_seqs), len(utr3_seqs), total_possible,
    )
    logger.info("Will process: %d combinations (chunk_size=%d, workers=%d)",
                actual_count, cfg.chunk_size, cfg.workers)

    if dry_run:
        print("\n" + "=" * 60)
        print("  DRY RUN — Input Validation & Planned Run Stats")
        print("=" * 60)
        print(f"\n  5' UTR sequences: {len(utr5_seqs)} (from {utr5_dir})")
        print(f"  ORF sequences:    {len(orf_seqs)} (from {orf_dir})")
        print(f"  3' UTR sequences: {len(utr3_seqs)} (from {utr3_dir})")
        print(f"\n  Total possible combinations: {total_possible:,}")
        print(f"  Combinations to process:     {actual_count:,}")
        print(f"  Chunk size:                  {cfg.chunk_size:,}")
        print(f"  Workers:                     {cfg.workers}")
        print(f"  Cap structure:               {cfg.cap}")
        print(f"  Kozak:                       {cfg.kozak}")
        print(f"  Poly-A tail length:          {cfg.polya_length} nt")
        print(f"  Codon optimization:          {'yes' if optimize else 'no'}")
        print(f"  Barcoding:                   {'skip' if no_barcode else cfg.barcoding.mode}")
        print(f"  Output directory:            {output}")
        print(f"\n  Scoring weights (normalized):")
        weights_dict = cfg.scoring_weights.to_dict()
        for k, v in weights_dict.items():
            print(f"    {k:20s} {v:.4f}")
        print("\n  ✓ Configuration valid. Ready to run.")
        print("=" * 60)
        return []

    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  mRNA Library Design Pipeline")
    print("=" * 60)

    # 1. (Optional) codon-optimize ORFs
    if optimize:
        t0 = time.time()
        logger.info("Stage 1/5: Optimizing ORF codon usage...")
        print("\n[1/5] Optimizing ORF codon usage...")
        optimize_sequences(orf_dir, cfg.optimizer.model_dump())
        timings["optimization"] = time.time() - t0
    else:
        print("\n[1/5] Skipping codon optimization (use --optimize to enable)")
        timings["optimization"] = 0.0

    # 2. Assemble in chunks
    t0 = time.time()
    logger.info("Stage 2/5: Assembling mRNA combinations...")
    print("\n[2/5] Assembling mRNA combinations...")

    weights = cfg.scoring_weights.to_dict()
    library: list[dict] = []
    chunk: list[dict] = []
    combo_idx = 0

    for u5, orf, u3 in _iter_combinations(utr5_seqs, orf_seqs, utr3_seqs, cfg.max_combinations):
        combo_idx += 1
        construct = assemble_construct(u5, orf, u3, cfg.cap, cfg.kozak, cfg.polya_length)
        construct["id"] = f"mRNA-{combo_idx:04d}"
        chunk.append(construct)

        if len(chunk) >= cfg.chunk_size:
            # Score this chunk
            scored = _score_chunk_with_transparency(chunk, weights)
            library.extend(scored)
            chunk = []

    # Score remaining
    if chunk:
        scored = _score_chunk_with_transparency(chunk, weights)
        library.extend(scored)

    timings["assembly_scoring"] = time.time() - t0
    print(f"    → {len(library)} combinations assembled and scored")
    logger.info("Assembly+scoring complete: %d constructs in %.2fs", len(library), timings["assembly_scoring"])

    # 3. Sort by composite score
    library.sort(key=lambda x: x["composite_score"], reverse=True)
    if library:
        print(f"    → Top score: {library[0]['composite_score']:.3f}")

    # 4. Barcode
    if not no_barcode:
        t0 = time.time()
        logger.info("Stage 3/5: Assigning barcodes...")
        print("\n[3/5] Assigning barcodes...")
        library = assign_barcodes(library, bc_cfg=cfg.barcoding.model_dump())
        timings["barcoding"] = time.time() - t0
        print(f"    → {len(library)} unique barcodes assigned")
    else:
        print("\n[3/5] Skipping barcoding (--no-barcode set)")
        timings["barcoding"] = 0.0

    # 5. Quality control
    t0 = time.time()
    logger.info("Stage 4/5: Running sequence QC checks...")
    print("\n[4/5] Running sequence QC checks...")
    library = qc_library(library, config=cfg.qc.model_dump())
    timings["qc"] = time.time() - t0
    passed = sum(1 for c in library if c.get("qc_passed", True))
    warnings = sum(c.get("qc_warnings", 0) for c in library)
    critical = sum(c.get("qc_critical", 0) for c in library)
    print(f"    → {passed}/{len(library)} passed QC")
    if warnings:
        print(f"    → {warnings} total warning(s)")
    if critical:
        print(f"    ⚠ {critical} critical issue(s) found")

    # Re-sort by adjusted score if available
    if library and "composite_score_adjusted" in library[0]:
        library.sort(key=lambda x: x.get("composite_score_adjusted", 0), reverse=True)
    else:
        library.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    # 6. Write outputs
    t0 = time.time()
    logger.info("Stage 5/5: Writing outputs...")
    print("\n[5/5] Writing outputs...")
    _write_outputs(library, output)
    if explain_top and library:
        _write_explain_top(library[:explain_top], output, weights)
    timings["output"] = time.time() - t0

    # Print timing summary
    total_time = sum(timings.values())
    print(f"\n✓ Done. Outputs written to: {output}/")
    logger.info("Pipeline complete in %.2fs", total_time)
    print(f"\n  Stage timings:")
    for stage, elapsed in timings.items():
        print(f"    {stage:20s} {elapsed:.3f}s")
    print(f"    {'TOTAL':20s} {total_time:.3f}s")
    print("=" * 60)

    return library


def _score_chunk_with_transparency(
    chunk: list[dict],
    weights: dict[str, float],
) -> list[dict]:
    """Score a chunk and add transparency fields (per-metric raw, normalized, weighted)."""
    scored_dims = [
        "cai", "mfe_stability", "gc_score", "utr_access",
        "codon_pair", "u_depletion", "cpg_depletion",
        "translation_eff", "codon_ramp", "codon_diversity",
    ]
    total_weight = sum(weights.get(k, 0) for k in scored_dims)

    results = []
    for construct in chunk:
        scored = score_construct(construct, weights)
        # Add transparency: per-metric weighted contribution
        for dim in scored_dims:
            raw_score = scored.get(dim, 0.0)
            w = weights.get(dim, 0.0)
            contribution = (w * raw_score / total_weight) if total_weight > 0 else 0.0
            scored[f"{dim}_weight"] = round(w, 6)
            scored[f"{dim}_contribution"] = round(contribution, 6)
        results.append(scored)
    return results


def _write_outputs(library: list[dict], output: Path) -> None:
    """Write CSV, FASTA, and JSON outputs."""
    # CSV (ranked)
    csv_path = output / "library_ranked.csv"
    if library:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(library[0].keys()))
            writer.writeheader()
            writer.writerows(library)

    # FASTA of full sequences
    fasta_path = output / "library.fasta"
    with open(fasta_path, "w") as f:
        for entry in library:
            header = (
                f">{entry['id']} "
                f"utr5={entry['utr5_name']} "
                f"orf={entry['orf_name']} "
                f"utr3={entry['utr3_name']} "
                f"score={entry['composite_score']:.4f} "
                f"barcode={entry.get('barcode', 'N/A')}"
            )
            f.write(header + "\n")
            f.write(entry["full_sequence"] + "\n")

    # JSON for downstream analysis
    json_path = output / "library.json"
    with open(json_path, "w") as f:
        json.dump(library, f, indent=2)

    print(f"    → {csv_path}")
    print(f"    → {fasta_path}")
    print(f"    → {json_path}")


def _write_explain_top(
    top_constructs: list[dict],
    output: Path,
    weights: dict[str, float],
) -> None:
    """Write a human-readable explanation of why top constructs ranked highest."""
    scored_dims = [
        "cai", "mfe_stability", "gc_score", "utr_access",
        "codon_pair", "u_depletion", "cpg_depletion",
        "translation_eff", "codon_ramp", "codon_diversity",
    ]

    explain_path = output / "explain_top.txt"
    with open(explain_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  Top {len(top_constructs)} Construct Explanations\n")
        f.write("=" * 70 + "\n\n")

        for rank, construct in enumerate(top_constructs, 1):
            f.write(f"Rank #{rank}: {construct['id']}\n")
            f.write(f"  Components: 5'UTR={construct['utr5_name']}, "
                    f"ORF={construct['orf_name']}, 3'UTR={construct['utr3_name']}\n")
            f.write(f"  Composite Score: {construct['composite_score']:.4f}\n")
            f.write(f"  Length: {construct['total_length']} nt\n\n")
            f.write(f"  {'Metric':<20s} {'Raw Score':>10s} {'Weight':>8s} {'Contribution':>14s}\n")
            f.write(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*14}\n")

            for dim in scored_dims:
                raw = construct.get(dim, 0.0)
                w = weights.get(dim, 0.0)
                contrib = construct.get(f"{dim}_contribution", 0.0)
                f.write(f"  {dim:<20s} {raw:>10.4f} {w:>8.4f} {contrib:>14.4f}\n")

            f.write("\n" + "-" * 70 + "\n\n")

    print(f"    → {explain_path}")
