# mRNA Library Design

A Python toolkit for designing, scoring, and barcoding combinatorial mRNA libraries.
Given FASTA files for 5' UTRs, ORFs, and 3' UTRs, it generates every combination,
scores each construct on CAI, MFE stability, GC content, and UTR accessibility,
assigns unique error-correcting barcodes, and outputs ranked FASTA + CSV.

![CI](https://github.com/YOUR_USERNAME/mrna-library-design/actions/workflows/ci.yml/badge.svg)

---

## Features

- Combinatorial assembly: 5'UTR × ORF × 3'UTR with configurable cap, Kozak, and poly-A
- Multi-objective scoring (CAI, MFE proxy, GC content, UTR accessibility)
- Error-correcting barcodes: DNA, peptide-encoding (LC-MS/MS), or QuART retron style
- Built-in codon optimizer with hooks for VaxPress / LinearDesign
- Ranked CSV + FASTA + JSON outputs
- GitHub Actions CI across Python 3.10 / 3.11 / 3.12

---

## Quick start

### 1. Check Python version

```bash
python3 --version
```
Python 3.10 or later is required.

### 2. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/mrna-library-design.git
cd mrna-library-design
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the web app (Streamlit)

```bash
streamlit run app.py
```

### 6. Or run the CLI pipeline on example data

```bash
python main.py \
  --utr5 data/utr5 \
  --orf  data/orf  \
  --utr3 data/utr3
```

Outputs land in `outputs/`:

| File | Contents |
|---|---|
| `library_ranked.csv` | All combinations, ranked by composite score |
| `library.fasta` | Full mRNA sequences with metadata headers |
| `library.json` | Machine-readable full results |

---

## Project structure

```
mrna-library-design/
├── mrna_design/
│   ├── assembler.py     # FASTA parsing + combinatorial assembly
│   ├── scorer.py        # CAI, MFE, GC, UTR scoring
│   ├── barcode.py       # Error-correcting barcode generation
│   └── optimizer.py     # Codon optimization (built-in + external hooks)
├── data/
│   ├── utr5/            # 5' UTR FASTA files
│   ├── orf/             # ORF / CDS FASTA files
│   └── utr3/            # 3' UTR FASTA files
├── tests/
│   └── test_mrna_design.py
├── outputs/             # Generated files (git-ignored)
├── notebooks/           # Jupyter analysis notebooks
├── main.py              # CLI entry point
├── config.yaml          # All parameters
├── requirements.txt
└── environment.yml      # Conda environment
```

---

## Configuration

Edit `config.yaml` to set paths, assembly parameters, scoring weights, and barcode settings.
All options can also be passed as CLI flags (which override config):

```bash
python main.py --max-combos 500 --optimize --output results/
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Upgrading scoring to real models

The proxy scoring functions in `mrna_design/scorer.py` are drop-in replaceable:

| Metric | Replacement |
|---|---|
| CAI | `Bio.SeqUtils.CodonAdaptationIndex` from BioPython |
| MFE | ViennaRNA Python bindings (`import RNA`) |
| Translation efficiency | mRNABERT / mRNA-LM embeddings |
| Full joint optimization | VaxPress or LinearDesign |

---

## License

MIT
