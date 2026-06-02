#!/usr/bin/env python3
"""
mRNA Library Design — main CLI entry point.

Usage:
    python main.py --utr5 data/utr5/ --orf data/orf/ --utr3 data/utr3/
    python main.py --config config.yaml
    python main.py --help
"""
import argparse
import sys
from pathlib import Path

import yaml

from mrna_design.assembler import assemble_library
from mrna_design.scorer import score_library
from mrna_design.barcode import assign_barcodes
from mrna_design.optimizer import optimize_sequences
from mrna_design.qc import qc_library


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate, score, and barcode a combinatorial mRNA library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--utr5", type=Path, help="Directory of 5' UTR FASTA files")
    parser.add_argument("--orf", type=Path, help="Directory of ORF/CDS FASTA files")
    parser.add_argument("--utr3", type=Path, help="Directory of 3' UTR FASTA files")
    parser.add_argument("--output", type=Path, default=Path("outputs"),
                        help="Output directory")
    parser.add_argument("--max-combos", type=int, default=None,
                        help="Cap on total combinations (overrides config)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run codon optimization on ORF sequences before assembly")
    parser.add_argument("--no-barcode", action="store_true",
                        help="Skip barcode assignment")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI args override config values
    utr5_dir  = args.utr5  or Path(cfg.get("utr5_dir",  "data/utr5"))
    orf_dir   = args.orf   or Path(cfg.get("orf_dir",   "data/orf"))
    utr3_dir  = args.utr3  or Path(cfg.get("utr3_dir",  "data/utr3"))
    output    = args.output or Path(cfg.get("output_dir", "outputs"))
    max_combos = args.max_combos or cfg.get("max_combinations", 1000)

    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  mRNA Library Design Pipeline")
    print("=" * 60)

    # 1. (Optional) codon-optimize ORFs
    if args.optimize:
        print("\n[1/4] Optimizing ORF codon usage...")
        optimize_sequences(orf_dir, cfg.get("optimizer", {}))
    else:
        print("\n[1/4] Skipping codon optimization (use --optimize to enable)")

    # 2. Assemble combinatorial library
    print("\n[2/4] Assembling mRNA combinations...")
    library = assemble_library(
        utr5_dir=utr5_dir,
        orf_dir=orf_dir,
        utr3_dir=utr3_dir,
        cap=cfg.get("cap", "m7G"),
        kozak=cfg.get("kozak", "strong"),
        polya_len=cfg.get("polya_length", 100),
        max_combinations=max_combos,
    )
    print(f"    → {len(library)} combinations assembled")

    # 3. Score
    print("\n[3/4] Scoring combinations...")
    library = score_library(library, weights=cfg.get("scoring_weights", {}))
    library.sort(key=lambda x: x["composite_score"], reverse=True)
    print(f"    → Top score: {library[0]['composite_score']:.3f}")

    # 4. Barcode
    if not args.no_barcode:
        print("\n[4/5] Assigning barcodes...")
        library = assign_barcodes(library, bc_cfg=cfg.get("barcoding", {}))
        print(f"    → {len(library)} unique barcodes assigned")
    else:
        print("\n[4/5] Skipping barcoding (--no-barcode set)")

    # 5. Quality control
    print("\n[5/5] Running sequence QC checks...")
    library = qc_library(library, config=cfg.get("qc", {}))
    passed = sum(1 for c in library if c.get("qc_passed", True))
    warnings = sum(c.get("qc_warnings", 0) for c in library)
    critical = sum(c.get("qc_critical", 0) for c in library)
    print(f"    → {passed}/{len(library)} passed QC")
    if warnings:
        print(f"    → {warnings} total warning(s)")
    if critical:
        print(f"    ⚠ {critical} critical issue(s) found")

    # Re-sort by adjusted score if available
    sort_key = "composite_score_adjusted" if "composite_score_adjusted" in library[0] else "composite_score"
    library.sort(key=lambda x: x.get(sort_key, 0), reverse=True)

    # Write outputs
    _write_outputs(library, output)
    print(f"\n✓ Done. Outputs written to: {output}/")
    print("=" * 60)


def _write_outputs(library: list[dict], output: Path):
    import csv, json

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
                f"barcode={entry.get('barcode','N/A')}"
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


if __name__ == "__main__":
    main()
