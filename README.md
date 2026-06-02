# mRNA Library Design

A Python toolkit for designing, scoring, and barcoding combinatorial mRNA libraries.
Given FASTA files for 5' UTRs, ORFs, and 3' UTRs, it generates every combination,
scores each construct using biophysically-grounded models (CAI, MFE, codon pair bias,
immunogenicity metrics), runs quality control checks, assigns unique error-correcting
barcodes, and outputs ranked FASTA + CSV.

![CI](https://github.com/YOUR_USERNAME/mrna-library-design/actions/workflows/ci.yml/badge.svg)

---

## Features

- Combinatorial assembly: 5'UTR × ORF × 3'UTR with configurable cap, Kozak, and poly-A
- **Multi-objective scoring** with 10 biophysical metrics:
  - CAI (geometric mean of relative adaptiveness — Sharp & Li method)
  - MFE stability (ViennaRNA if installed, else nearest-neighbour dinucleotide model)
  - GC content optimality (Gaussian penalty model)
  - UTR accessibility (structure + motif-aware: TOP, ARE, CPE detection)
  - Codon pair bias (comprehensive CPB table — Coleman et al. 2008)
  - Uridine depletion (innate immune evasion for modified mRNA)
  - CpG depletion (TLR9 avoidance)
  - Translation efficiency (5'UTR k-mer model inspired by Optimus 5-Prime)
  - Codon ramp (5' ORF slow ramp for co-translational folding — Tuller et al.)
  - Codon diversity (autocorrelation avoidance — tRNA pool depletion)
- **Sequence QC module** with 9 automated checks:
  - CpG dinucleotide density (immunogenicity)
  - Uridine content (immune activation)
  - Homopolymer runs (synthesis failure risk)
  - Local GC extremes (folding/synthesis issues)
  - Restriction enzyme sites (cloning compatibility)
  - Premature polyadenylation signals
  - Upstream AUG in 5'UTR (leaky scanning risk)
  - Inverted repeats / dsRNA potential (innate immune sensors)
  - miRNA seed matches in 3'UTR (stability risk)
- **Multi-objective codon optimizer** (balanced CAI + low-U + low-CpG + pair bias + autocorrelation)
- Error-correcting barcodes: DNA, peptide-encoding (LC-MS/MS), or QuART retron style
- Built-in codon optimizer with hooks for VaxPress / LinearDesign
- Ranked CSV + FASTA + JSON outputs
- Streamlit web interface with interactive visualisations
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
│   ├── scorer.py        # CAI, MFE, GC, UTR, codon pair, immunogenicity scoring
│   ├── barcode.py       # Error-correcting barcode generation
│   ├── optimizer.py     # Multi-objective codon optimization
│   └── qc.py           # Sequence quality control (9 automated checks)
├── data/
│   ├── utr5/            # 5' UTR FASTA files
│   ├── orf/             # ORF / CDS FASTA files
│   └── utr3/            # 3' UTR FASTA files
├── tests/
│   └── test_mrna_design.py
├── outputs/             # Generated files (git-ignored)
├── notebooks/           # Jupyter analysis notebooks
├── main.py              # CLI entry point
├── app.py               # Streamlit web interface
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

## Scoring models

The scoring pipeline in `mrna_design/scorer.py` uses 10 biophysical metrics.
Below is what each implements and what optional deep-learning upgrades exist:

| Metric | Current Implementation | Optional Upgrade |
|---|---|---|
| CAI | Geometric mean of relative adaptiveness (Sharp & Li) | — (already production-grade) |
| MFE | ViennaRNA `RNA.fold()` or dinucleotide NN model (Turner 2004) | — (already realistic) |
| GC optimality | Gaussian penalty centered at 52% (σ=0.12) | — |
| UTR accessibility | Structure + motif-aware (TOP, ARE, CPE, near-cognate starts) | mRNABERT / UTR-LM embeddings |
| Codon pair bias | Comprehensive CPB table (100+ pairs, Coleman et al. 2008) | — (already comprehensive) |
| Uridine depletion | Gaussian penalty at 17% U (σ=0.05) | — |
| CpG depletion | Linear density penalty (target < 0.005) | — |
| Translation efficiency | 5'UTR k-mer model (Optimus 5-Prime inspired) + G4/uORF detection | Saluki, iCodon, or deep-learning TE models |
| Codon ramp | 5' ORF CAI gradient (Tuller et al. 2010, Hanson & Coller 2018) | Ribosome profiling calibration |
| Codon diversity | Autocorrelation + synonym usage + junction diversity | — |

The codon optimizer (`mrna_design/optimizer.py`) balances 5 objectives per codon:
- CAI (40%), Uridine depletion (25%), CpG avoidance (15%),
  Autocorrelation avoidance (10%), Codon pair bias (10%)

To enable ViennaRNA:
```bash
pip install ViennaRNA
# or: conda install -c bioconda viennarna
```

---

## License

MIT
