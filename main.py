#!/usr/bin/env python3
"""
mRNA Library Design — main CLI entry point.

Usage:
    python main.py --utr5 data/utr5/ --orf data/orf/ --utr3 data/utr3/
    python main.py --config config.yaml
    python main.py --dry-run
    python main.py --chunk-size 2000 --workers 4
    python main.py --explain-top 10
    python main.py --help
"""
import argparse
import logging
import sys
from pathlib import Path

from mrna_design.config_model import load_and_validate_config
from mrna_design.pipeline import run_pipeline


def parse_args():
    """Parse CLI arguments for the mRNA library design pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate, score, and barcode a combinatorial mRNA library.\n\n"
            "Examples:\n"
            "  python main.py --config config.yaml\n"
            "  python main.py --utr5 data/utr5 --orf data/orf --utr3 data/utr3\n"
            "  python main.py --dry-run                    # validate without running\n"
            "  python main.py --chunk-size 2000 --workers 4  # parallel processing\n"
            "  python main.py --explain-top 10             # explain top 10 rankings\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--config", type=Path, default=Path("config.yaml"),
                          help="Path to YAML config file (default: config.yaml)")
    io_group.add_argument("--utr5", type=Path,
                          help="Directory of 5' UTR FASTA files (overrides config)")
    io_group.add_argument("--orf", type=Path,
                          help="Directory of ORF/CDS FASTA files (overrides config)")
    io_group.add_argument("--utr3", type=Path,
                          help="Directory of 3' UTR FASTA files (overrides config)")
    io_group.add_argument("--output", type=Path,
                          help="Output directory (overrides config)")

    # Pipeline options
    pipe_group = parser.add_argument_group("Pipeline Options")
    pipe_group.add_argument("--max-combos", type=int, default=None,
                            help="Cap on total combinations (overrides config)")
    pipe_group.add_argument("--optimize", action="store_true",
                            help="Run codon optimization on ORF sequences before assembly")
    pipe_group.add_argument("--no-barcode", action="store_true",
                            help="Skip barcode assignment")

    # Performance
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--chunk-size", type=int, default=None,
                            help="Number of constructs to process per chunk (default: 5000)")
    perf_group.add_argument("--workers", type=int, default=None,
                            help="Number of parallel scoring workers (default: 1)")

    # Transparency
    trans_group = parser.add_argument_group("Scoring Transparency")
    trans_group.add_argument("--explain-top", type=int, default=None, metavar="N",
                            help="Generate explanation artifact for top N constructs")

    # Developer ergonomics
    dev_group = parser.add_argument_group("Developer Options")
    dev_group.add_argument("--dry-run", action="store_true",
                           help="Validate inputs/config and print planned run stats without executing")
    dev_group.add_argument("--verbose", "-v", action="store_true",
                           help="Enable verbose/debug logging with stage timings")

    return parser.parse_args()


def main():
    """Entry point for the mRNA library design CLI."""
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("mrna_design")

    # Build config overrides from CLI args
    overrides: dict = {}
    if args.utr5:
        overrides["utr5_dir"] = str(args.utr5)
    if args.orf:
        overrides["orf_dir"] = str(args.orf)
    if args.utr3:
        overrides["utr3_dir"] = str(args.utr3)
    if args.output:
        overrides["output_dir"] = str(args.output)
    if args.max_combos is not None:
        overrides["max_combinations"] = args.max_combos
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.workers is not None:
        overrides["workers"] = args.workers

    # Load and validate config
    try:
        cfg = load_and_validate_config(args.config, overrides)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.debug("Configuration loaded and validated successfully")

    # Run pipeline
    try:
        run_pipeline(
            cfg=cfg,
            optimize=args.optimize,
            no_barcode=args.no_barcode,
            explain_top=args.explain_top,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
