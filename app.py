"""
app.py — Streamlit web interface for mRNA Library Designer.

Upload 5' UTR, ORF, and 3' UTR FASTA files · Score · Barcode · Download
"""
import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from mrna_design.assembler import (
    assemble_library,
    load_fasta_dir,
    read_fasta,
)
from mrna_design.scorer import score_library
from mrna_design.barcode import assign_barcodes

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="mRNA Library Designer",
    page_icon="🧬",
    layout="wide",
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("5' Cap")
    cap_type = st.selectbox(
        "5' Cap",
        options=["m7G", "cap1", "arca", "none"],
        index=0,
        label_visibility="collapsed",
    )

    st.header("Kozak context")
    kozak = st.selectbox(
        "Kozak context",
        options=["strong", "moderate", "none"],
        index=0,
        label_visibility="collapsed",
    )

    st.header("Poly-A tail length (nt)")
    polya_len = st.slider(
        "Poly-A tail length (nt)",
        min_value=0,
        max_value=300,
        value=100,
        label_visibility="collapsed",
    )

    st.header("Max combinations")
    max_combinations = st.number_input(
        "Max combinations",
        min_value=1,
        max_value=100000,
        value=500,
        step=1,
        label_visibility="collapsed",
    )

    st.divider()

    st.header("📊 Scoring weights")

    cai_weight = st.slider("CAI weight", 0.0, 1.0, 0.30, 0.01)
    mfe_weight = st.slider("MFE stability weight", 0.0, 1.0, 0.25, 0.01)
    gc_weight = st.slider("GC content weight", 0.0, 1.0, 0.20, 0.01)
    utr_weight = st.slider("UTR accessibility", 0.0, 1.0, 0.25, 0.01)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🧬 mRNA Library Designer")
st.caption("Upload 5' UTR, ORF, and 3' UTR FASTA files · Score · Barcode · Download")

st.header("1. Upload FASTA files")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("5' UTR sequences")
    utr5_files = st.file_uploader(
        "Upload 5' UTR FASTA",
        accept_multiple_files=True,
        type=["fa", "fasta", "txt"],
        key="utr5",
    )

with col2:
    st.subheader("ORF / CDS sequences")
    orf_files = st.file_uploader(
        "Upload ORF FASTA",
        accept_multiple_files=True,
        type=["fa", "fasta", "txt"],
        key="orf",
    )

with col3:
    st.subheader("3' UTR sequences")
    utr3_files = st.file_uploader(
        "Upload 3' UTR FASTA",
        accept_multiple_files=True,
        type=["fa", "fasta", "txt"],
        key="utr3",
    )

use_examples = st.checkbox("Use built-in example sequences", value=True)

# ── Generate library ──────────────────────────────────────────────────────────
st.header("2. Generate library")

if st.button("🧪 Generate mRNA Library", type="primary", use_container_width=True):
    with st.spinner("Generating mRNA library..."):
        # Determine data directories
        if use_examples and not (utr5_files and orf_files and utr3_files):
            # Use built-in example data
            base = Path(__file__).parent / "data"
            utr5_dir = base / "utr5"
            orf_dir = base / "orf"
            utr3_dir = base / "utr3"
        else:
            # Write uploaded files to temporary directories
            tmpdir = Path(tempfile.mkdtemp())
            utr5_dir = tmpdir / "utr5"
            orf_dir = tmpdir / "orf"
            utr3_dir = tmpdir / "utr3"
            utr5_dir.mkdir()
            orf_dir.mkdir()
            utr3_dir.mkdir()

            for f in (utr5_files or []):
                (utr5_dir / f.name).write_bytes(f.read())
            for f in (orf_files or []):
                (orf_dir / f.name).write_bytes(f.read())
            for f in (utr3_files or []):
                (utr3_dir / f.name).write_bytes(f.read())

        try:
            # Assemble
            library = assemble_library(
                utr5_dir=utr5_dir,
                orf_dir=orf_dir,
                utr3_dir=utr3_dir,
                cap=cap_type,
                kozak=kozak,
                polya_len=polya_len,
                max_combinations=int(max_combinations),
            )

            # Score
            weights = {
                "cai": cai_weight,
                "mfe_stability": mfe_weight,
                "gc_content": gc_weight,
                "utr_access": utr_weight,
            }
            scored = score_library(library, weights)

            # Barcode
            barcoded = assign_barcodes(scored)

            # Display results
            df = pd.DataFrame(barcoded)
            df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

            st.success(f"✅ Generated {len(df)} mRNA constructs")

            st.subheader("Top constructs")
            display_cols = [
                "id", "utr5_name", "orf_name", "utr3_name",
                "composite_score", "cai", "mfe_stability", "gc_content",
                "utr_access", "barcode", "total_length",
            ]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols].head(20), use_container_width=True)

            # Download CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download full library (CSV)",
                data=csv_buffer.getvalue(),
                file_name="mrna_library.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
