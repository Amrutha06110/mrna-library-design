"""
app.py — Streamlit web interface for mRNA Library Designer.

Fully interactive mRNA library designer with barcode generation,
construct assembly, scoring, and export.
"""
import io
import random
import string

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="mRNA Library Designer",
    page_icon="🧬",
    layout="wide",
)

# ── Human codon usage table (fraction of max per amino acid) ──────────────────
HUMAN_CODON_TABLE = {
    "TTT": 0.45, "TTC": 1.00, "TTA": 0.07, "TTG": 0.13,
    "CTT": 0.13, "CTC": 0.20, "CTA": 0.07, "CTG": 1.00,
    "ATT": 0.36, "ATC": 1.00, "ATA": 0.07, "ATG": 1.00,
    "GTT": 0.18, "GTC": 0.24, "GTA": 0.07, "GTG": 1.00,
    "TCT": 0.15, "TCC": 0.22, "TCA": 0.12, "TCG": 0.06,
    "AGT": 0.15, "AGC": 1.00, "CCT": 0.28, "CCC": 1.00,
    "CCA": 0.27, "CCG": 0.11, "ACT": 0.24, "ACC": 1.00,
    "ACA": 0.28, "ACG": 0.12, "GCT": 0.26, "GCC": 1.00,
    "GCA": 0.23, "GCG": 0.11, "TAT": 0.43, "TAC": 1.00,
    "TAA": 0.28, "TAG": 0.20, "TGA": 1.00, "CAT": 0.41,
    "CAC": 1.00, "CAA": 0.25, "CAG": 1.00, "AAT": 0.46,
    "AAC": 1.00, "AAA": 0.43, "AAG": 1.00, "GAT": 0.46,
    "GAC": 1.00, "GAA": 0.42, "GAG": 1.00, "TGT": 0.45,
    "TGC": 1.00, "TGG": 1.00, "CGT": 0.08, "CGC": 0.19,
    "CGA": 0.11, "CGG": 0.20, "AGA": 0.20, "AGG": 1.00,
    "GGT": 0.16, "GGC": 1.00, "GGA": 0.25, "GGG": 0.25,
}

# ── 3'UTR sequences ──────────────────────────────────────────────────────────
UTR3_SEQUENCES = {
    "Human beta-globin": (
        "GCTCGCTTTCTTGCTGTCCAATTTCTATTAAAGGTTCCTTTGTTCCCTAAGTCCAACTACTAAACTGG"
        "GGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGC"
    ),
    "Human alpha-globin": (
        "GCTGGAGCCTCGGTGGCCATGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCA"
        "CCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGC"
    ),
    "AES + mtRNR1 (Pfizer-style)": (
        "CTGATAATGATTTTATTTTGACTGATAGTGACCTGTTCGTTGCAACAAATTGATGAGCAATGCTTTTTT"
        "ATAATGCCAACTTTGTACAAAAAAGCAGGCTTTAAAGGAACCAATTCAGTCGACTGGATCCGGTACCGAA"
        "TTCGATATCAAGCTTATCGATACCGTCGACCTCGAGGGGGGGCCCGGTACCCAATTCGCCCTATAGTGAG"
        "TCGTATTACAATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTTACCCAACTTAA"
        "TCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCC"
        "AACAGTTGCGCAGCCTGAATGGCGAATG"
    ),
    "Human albumin": (
        "CATCACATTTAAAAGCATCTCAGCCTACCATGAGAATAAGAGAAAGAAAATGAAGATCAAAAGCTTATTCA"
        "TCTGTTTTTCTTTTTCGTTGGTGTAAAGCCAACACCCTGTCTAAAAAACATAAATTTCTTTAATCATTTTG"
        "CCTCTTTTCTCTGTGCTTCAATTAATAAAAAATGGAAAGAACCTCGAG"
    ),
}

# ── 5'UTR sequences ──────────────────────────────────────────────────────────
UTR5_SEQUENCES = {
    "Human beta-globin": "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACCATG",
    "Kozak consensus": "GCCACCATG",
    "HSP70": "AGCAAAAGCAGGTAGATATTGAAAGAT",
    "Tobacco mosaic virus": "GTATTTTACAACAATTACCAACAACAACAAACAACAAACAACATTACAATTACTATTTACAATTACA",
}

UTR5_HELP = {
    "Human beta-globin": "Widely used in mRNA therapeutics; enhances translation",
    "Kozak consensus": "Minimal strong Kozak sequence for efficient initiation",
    "HSP70": "Heat shock protein 70 UTR; stress-responsive enhancement",
    "Tobacco mosaic virus": "Viral omega leader; strong cap-independent translation",
}

UTR3_HELP = {
    "Human beta-globin": "Standard 3'UTR for mRNA stability; used in many vaccines",
    "Human alpha-globin": "Enhances mRNA stability and half-life",
    "AES + mtRNR1 (Pfizer-style)": "Dual UTR used in BNT162b2; high stability",
    "Human albumin": "Long half-life UTR from albumin mRNA",
    "Custom": "Paste your own 3'UTR sequence",
}

# ── Helper functions ──────────────────────────────────────────────────────────


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA/RNA sequence (as DNA)."""
    comp = {"A": "T", "T": "A", "U": "A", "G": "C", "C": "G",
            "a": "t", "t": "a", "u": "a", "g": "c", "c": "g"}
    return "".join(comp.get(b, b) for b in reversed(seq))


def hamming_distance(s1: str, s2: str) -> int:
    """Hamming distance between two equal-length strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def has_homopolymer(seq: str, max_run: int = 3) -> bool:
    """Check if sequence has homopolymer run > max_run."""
    count = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            count += 1
            if count > max_run:
                return True
        else:
            count = 1
    return False


def gc_content(seq: str) -> float:
    """Calculate GC content as fraction."""
    if not seq:
        return 0.0
    gc = sum(1 for b in seq.upper() if b in "GC")
    return gc / len(seq)


# Nearest-neighbour RNA dinucleotide stacking energies (kcal/mol, 37°C)
_DINUC_ENERGY = {
    "AA": -0.93, "AU": -1.10, "AG": -2.08, "AC": -2.24,
    "UA": -1.33, "UU": -0.93, "UG": -2.08, "UC": -2.11,
    "GA": -2.35, "GU": -2.11, "GG": -3.26, "GC": -3.42,
    "CA": -2.11, "CU": -2.08, "CG": -2.36, "CC": -3.26,
    "TA": -1.33, "TU": -0.93, "TG": -2.08, "TC": -2.11,
    "AT": -1.10, "GT": -2.11, "CT": -2.08, "TT": -0.93,
}


def _calculate_mfe_score(seq: str) -> float:
    """
    Physics-based MFE proxy using nearest-neighbour dinucleotide
    stacking energies. Returns 0–1 (1 = very stable).
    """
    if len(seq) < 2:
        return 0.5
    seq_upper = seq.upper()
    total_energy = 0.0
    count = 0
    for i in range(len(seq_upper) - 1):
        dinuc = seq_upper[i:i+2]
        e = _DINUC_ENERGY.get(dinuc, -1.5)
        total_energy += e
        count += 1
    if count == 0:
        return 0.5
    energy_per_nt = total_energy / count
    # Map range [-1.0, -3.5] to [0, 1]
    score = (abs(energy_per_nt) - 1.0) / 2.5
    return max(0.0, min(1.0, score))


def generate_barcodes(n: int, length: int, min_hamming: int,
                      gc_min: float, gc_max: float) -> list:
    """Generate n orthogonal barcodes satisfying constraints."""
    bases = "ACGT"
    barcodes = []
    max_attempts = n * 1000

    for _ in range(max_attempts):
        if len(barcodes) >= n:
            break
        # Generate random barcode
        bc = "".join(random.choice(bases) for _ in range(length))
        # Check GC content
        gc = gc_content(bc)
        if gc < gc_min or gc > gc_max:
            continue
        # Check homopolymer
        if has_homopolymer(bc, 3):
            continue
        # Check Hamming distance to all existing barcodes
        if all(hamming_distance(bc, existing) >= min_hamming for existing in barcodes):
            barcodes.append(bc)

    return barcodes


def calculate_cai(cds: str) -> float:
    """Calculate Codon Adaptation Index for a CDS."""
    cds = cds.upper().replace("U", "T")
    if len(cds) < 3:
        return 0.0
    codons = [cds[i:i+3] for i in range(0, len(cds) - 2, 3)]
    scores = []
    for codon in codons:
        if len(codon) == 3 and codon in HUMAN_CODON_TABLE:
            scores.append(HUMAN_CODON_TABLE[codon])
    if not scores:
        return 0.0
    # Geometric mean
    log_scores = [np.log(s) if s > 0 else np.log(0.01) for s in scores]
    return float(np.exp(np.mean(log_scores)))


def read_fasta_text(text: str) -> list:
    """Parse FASTA format text, return list of (name, sequence) tuples."""
    sequences = []
    current_name = ""
    current_seq = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_name:
                sequences.append((current_name, "".join(current_seq)))
            current_name = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line)
    if current_name:
        sequences.append((current_name, "".join(current_seq)))
    return sequences


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Design Parameters")

    # ── Construct Settings ──
    st.markdown("### 🧪 Construct Settings")

    cap_type = st.selectbox(
        "5' Cap",
        options=["CleanCap AG", "m7G", "ARCA", "Cap1"],
        index=0,
    )

    kozak = st.selectbox(
        "Kozak context",
        options=["strong (GCCACCATG)", "moderate (ACCATG)", "weak (ATG)"],
        index=0,
    )

    utr5_choice = st.selectbox(
        "5' UTR",
        options=list(UTR5_SEQUENCES.keys()),
        index=0,
        help=UTR5_HELP.get("Human beta-globin", ""),
    )
    st.caption(UTR5_HELP.get(utr5_choice, ""))

    utr3_choice = st.selectbox(
        "3' UTR",
        options=list(UTR3_HELP.keys()),
        index=0,
        help=UTR3_HELP.get("Human beta-globin", ""),
    )
    st.caption(UTR3_HELP.get(utr3_choice, ""))

    custom_utr3_seq = ""
    if utr3_choice == "Custom":
        custom_utr3_seq = st.text_area(
            "Custom 3' UTR sequence",
            value="",
            placeholder="Paste your 3'UTR sequence here...",
        )

    polya_len = st.slider(
        "Poly-A tail length (nt)",
        min_value=0,
        max_value=200,
        value=100,
    )

    add_invdc = st.toggle("Add invdC at 3' end (via ligation)", value=True)
    if add_invdc:
        st.caption(
            "A pre-synthesized oligo with /3InvdC/ will be ligated "
            "using T4 RNA Ligase 1 after IVT"
        )

    st.divider()

    # ── Barcode Settings ──
    st.markdown("### 🧬 Barcode Settings")

    num_barcodes = st.slider("Number of barcodes", 1, 500, 96)
    barcode_length = st.slider("Barcode length (nt)", 10, 30, 20)
    min_hamming_dist = st.slider("Min Hamming distance", 2, 10, 4)
    gc_min_pct = st.slider("GC content min %", 30, 50, 40)
    gc_max_pct = st.slider("GC content max %", 50, 70, 60)
    barcode_position = st.selectbox(
        "Barcode position",
        options=["3'UTR start", "3'UTR end"],
        index=0,
    )

    st.divider()

    # ── Scoring Weights ──
    st.markdown("### 📊 Scoring Weights")

    cai_weight = st.slider("CAI weight", 0.0, 1.0, 0.30, 0.01)
    mfe_weight = st.slider("MFE stability weight", 0.0, 1.0, 0.25, 0.01)
    gc_weight = st.slider("GC content weight", 0.0, 1.0, 0.20, 0.01)
    utr_weight = st.slider("UTR accessibility weight", 0.0, 1.0, 0.25, 0.01)
    max_combinations = st.number_input("Max combinations", min_value=1, value=500, step=1)

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🧬 mRNA Library Designer")

# ── Step 1: Enter or use example sequences ──
st.header("Step 1: Enter sequences")

col1, col2, col3 = st.columns(3)

with col1:
    five_utr_input = st.text_area(
        "5' UTR sequence",
        value="",
        height=150,
        placeholder='>my_5utr\nACATTTGCTTC..." or paste raw sequence',
        key="utr5",
    )

with col2:
    cds_input = st.text_area(
        "ORF / CDS sequence",
        value="",
        height=150,
        placeholder='>my_cds\nATGGCCTTACC..." or paste raw sequence starting with ATG',
        key="orf",
    )

with col3:
    three_utr_input = st.text_area(
        "3' UTR sequence",
        value="",
        height=150,
        placeholder='>my_3utr\nUGCAAATAAAG..." or paste raw sequence',
        key="utr3",
    )

# ── Helper: parse pasted text (FASTA or raw) ─────────────────────────────────


def _parse_input_sequence(text: str) -> list:
    """Parse pasted text as FASTA or raw sequence.

    If text starts with '>', treat as FASTA (strip header, join remaining lines).
    Otherwise treat entire text as a raw sequence.
    Strips whitespace, converts to uppercase, replaces U with T for DNA processing.
    """
    text = text.strip()
    if not text:
        return []
    if text.startswith(">"):
        entries = read_fasta_text(text)
        # Normalize each sequence
        return [
            (name, seq.replace(" ", "").replace("\t", "").upper().replace("U", "T"))
            for name, seq in entries
        ]
    # Raw sequence: strip whitespace, uppercase, U->T
    raw = "".join(text.split()).upper().replace("U", "T")
    return [("pasted_sequence", raw)]


# ── Step 2: Generate Library ──
st.header("Step 2: Generate Library")

if st.button("🧬 Generate mRNA Library", type="primary", use_container_width=True):
    # Validate CDS is provided
    if not cds_input.strip():
        st.error("Please paste a CDS sequence to continue.")
    else:
        progress_bar = st.progress(0, text="Initializing...")

        # Parse pasted sequences
        cds_seqs = _parse_input_sequence(cds_input)

        # Fallback to built-in human beta-globin for empty UTR fields
        if five_utr_input.strip():
            utr5_seqs = _parse_input_sequence(five_utr_input)
        else:
            utr5_seqs = [("human_beta_globin_5UTR", UTR5_SEQUENCES["Human beta-globin"])]

        if three_utr_input.strip():
            utr3_seqs = _parse_input_sequence(three_utr_input)
        else:
            utr3_seqs = [("human_beta_globin_3UTR", UTR3_SEQUENCES["Human beta-globin"])]

        if not cds_seqs:
            st.error("Could not parse CDS sequence. Please check your input.")
        else:
            progress_bar.progress(10, text="Generating barcodes...")

            # Generate barcodes
            barcodes = generate_barcodes(
                n=num_barcodes,
                length=barcode_length,
                min_hamming=min_hamming_dist,
                gc_min=gc_min_pct / 100.0,
                gc_max=gc_max_pct / 100.0,
            )

            if len(barcodes) < num_barcodes:
                st.warning(
                    f"Could only generate {len(barcodes)} barcodes "
                    f"(requested {num_barcodes}). Try relaxing constraints."
                )

            progress_bar.progress(30, text="Assembling constructs...")

            # Get selected UTR sequences from sidebar
            selected_utr5_seq = UTR5_SEQUENCES.get(utr5_choice, "GCCACCATG")
            if utr3_choice == "Custom":
                selected_utr3_seq = custom_utr3_seq
            else:
                selected_utr3_seq = UTR3_SEQUENCES.get(utr3_choice, UTR3_SEQUENCES["Human beta-globin"])

            # T7 promoter
            t7_promoter = "TAATACGACTCACTATA"
            poly_a = "A" * polya_len

            # Assemble constructs
            constructs = []
            total = min(len(barcodes), int(max_combinations))

            for i, barcode in enumerate(barcodes[:total]):
                # Use first CDS (or cycle through uploaded)
                cds_name, cds_seq = cds_seqs[i % len(cds_seqs)]

                # Assemble mRNA construct
                if barcode_position == "3'UTR start":
                    mrna_seq = selected_utr5_seq + cds_seq + barcode + selected_utr3_seq + poly_a
                else:
                    mrna_seq = selected_utr5_seq + cds_seq + selected_utr3_seq + barcode + poly_a

                # DNA template: T7 + reverse complement of mRNA (U→T already DNA)
                mrna_as_dna = mrna_seq.replace("U", "T").replace("u", "t")
                dna_template = t7_promoter + reverse_complement(mrna_as_dna)

                # Scores — use real scoring functions
                cai_score = calculate_cai(cds_seq)
                gc_score = gc_content(mrna_seq)
                length_score = 1.0 / (1.0 + len(mrna_seq) / 1000.0)  # shorter is better

                # Individual sub-scores (realistic calculations)
                score_cai = cai_score

                # GC score: Gaussian penalty centered at 52%
                gc_deviation = gc_score - 0.52
                score_gc = float(np.exp(-(gc_deviation ** 2) / (2 * 0.12 ** 2)))

                # MFE proxy: dinucleotide stacking energy model
                score_mfe = _calculate_mfe_score(mrna_seq)

                # UTR accessibility: penalize high GC in 5'UTR (blocks scanning)
                utr5_gc = gc_content(selected_utr5_seq)
                score_utr = float(np.exp(-((utr5_gc - 0.40) ** 2) / (2 * 0.08 ** 2)))

                # Composite score
                composite = (
                    cai_weight * score_cai +
                    gc_weight * score_gc +
                    mfe_weight * score_mfe +
                    utr_weight * score_utr
                )

                construct = {
                    "Name": f"Construct_{i+1:04d}",
                    "Barcode": barcode,
                    "5UTR": selected_utr5_seq,
                    "CDS": cds_seq,
                    "CDS_Name": cds_name,
                    "3UTR": selected_utr3_seq,
                    "3UTR_Name": utr3_choice,
                    "PolyA": poly_a,
                    "mRNA_Sequence": mrna_seq,
                    "DNA_Template": dna_template,
                    "T7_Promoter": t7_promoter,
                    "GC_Pct": round(gc_score * 100, 2),
                    "Length": len(mrna_seq),
                    "CAI_Score": round(cai_score, 4),
                    "Score_CAI": round(score_cai, 4),
                    "Score_MFE": round(score_mfe, 4),
                    "Score_GC": round(score_gc, 4),
                    "Score_UTR": round(score_utr, 4),
                    "Total_Score": round(composite, 4),
                    "Composite_Score": round(composite, 4),
                    "Cap": cap_type,
                    "Kozak": kozak,
                    "PolyA_Length": polya_len,
                    "Barcode_Position": barcode_position,
                }
                constructs.append(construct)

                # Update progress
                if (i + 1) % max(1, total // 10) == 0:
                    pct = 30 + int(60 * (i + 1) / total)
                    progress_bar.progress(pct, text=f"Assembling construct {i+1}/{total}...")

            progress_bar.progress(95, text="Finalizing...")

            # Store in session state
            st.session_state["library"] = constructs
            st.session_state["barcodes"] = barcodes
            st.session_state["add_invdc"] = add_invdc
            st.session_state["scoring_weights"] = {
                "cai_weight": cai_weight,
                "mfe_weight": mfe_weight,
                "gc_weight": gc_weight,
                "utr_weight": utr_weight,
            }

            progress_bar.progress(100, text="Done!")
            st.success(f"✅ Generated {len(constructs)} mRNA constructs with {len(barcodes)} barcodes")

# ── Step 3: Results ──
if "library" in st.session_state and st.session_state["library"]:
    st.header("Step 3: Results")

    df = pd.DataFrame(st.session_state["library"])

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Library Summary",
        "🔬 Sequence Viewer",
        "🧫 DNA Templates (for IVT)",
        "💊 invdC Ligation Oligos",
    ])

    # ── Tab 1: Library Summary ──
    with tab1:
        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Constructs", len(df))
        m2.metric("Mean GC%", f"{df['GC_Pct'].mean():.1f}%")
        m3.metric("Mean Length", f"{df['Length'].mean():.0f} nt")
        m4.metric("Barcodes Generated", len(st.session_state["barcodes"]))

        # ── Scoring Explanation Panel ──
        weights = st.session_state.get("scoring_weights", {
            "cai_weight": 0.30, "mfe_weight": 0.25, "gc_weight": 0.20, "utr_weight": 0.25,
        })
        w_cai = weights["cai_weight"]
        w_mfe = weights["mfe_weight"]
        w_gc = weights["gc_weight"]
        w_utr = weights["utr_weight"]

        with st.expander("📊 How constructs are scored"):
            sc1, sc2, sc3, sc4 = st.columns(4)

            # Helper for color indicator
            def _color_indicator(score, metric_type="default"):
                if metric_type == "gc":
                    if 0.45 <= score <= 0.55:
                        return "🟢"
                    elif 0.40 <= score < 0.45 or 0.55 < score <= 0.60:
                        return "🟡"
                    else:
                        return "🔴"
                else:
                    if score > 0.7:
                        return "🟢"
                    elif 0.4 <= score <= 0.7:
                        return "🟡"
                    else:
                        return "🔴"

            # Column 1 — CAI
            with sc1:
                mean_cai = df["Score_CAI"].mean() if "Score_CAI" in df.columns else 0
                st.markdown(f"**CAI Weight: {w_cai}**")
                st.markdown(
                    "Measures how well your CDS codons match human preferences. "
                    "Higher = ribosomes translate faster and more efficiently. "
                    "Most important for protein yield."
                )
                st.markdown(f"Mean score: {mean_cai:.3f} {_color_indicator(mean_cai)}")

            # Column 2 — MFE Stability
            with sc2:
                mean_mfe = df["Score_MFE"].mean() if "Score_MFE" in df.columns else 0
                st.markdown(f"**MFE Stability Weight: {w_mfe}**")
                st.markdown(
                    "Minimum Free Energy — how stable the mRNA secondary structure is. "
                    "More negative MFE = longer half-life in cells. "
                    "Important for therapeutic durability."
                )
                st.markdown(f"Mean score: {mean_mfe:.3f} {_color_indicator(mean_mfe)}")

            # Column 3 — GC Content
            with sc3:
                mean_gc = df["Score_GC"].mean() if "Score_GC" in df.columns else 0
                # For GC indicator, use GC_Pct (as fraction)
                mean_gc_frac = df["GC_Pct"].mean() / 100.0
                st.markdown(f"**GC Content Weight: {w_gc}**")
                st.markdown(
                    "Balance of G/C vs A/U bases. Optimal is 50–55% for mRNA therapeutics. "
                    "Too low = unstable. Too high = over-structured and poorly translated."
                )
                st.markdown(f"Mean score: {mean_gc:.3f} {_color_indicator(mean_gc_frac, 'gc')}")

            # Column 4 — UTR Accessibility
            with sc4:
                mean_utr = df["Score_UTR"].mean() if "Score_UTR" in df.columns else 0
                st.markdown(f"**UTR Accessibility Weight: {w_utr}**")
                st.markdown(
                    "How open the 5'UTR is for ribosome binding. "
                    "Higher = ribosomes can reach the start codon more efficiently "
                    "= better translation initiation."
                )
                st.markdown(f"Mean score: {mean_utr:.3f} {_color_indicator(mean_utr)}")

            # Formula display
            st.info(
                f"**Total Score** = (CAI × {w_cai}) + (MFE × {w_mfe}) + (GC × {w_gc}) + (UTR × {w_utr})"
            )

            # Note box
            st.warning(
                "For LNP screening libraries, keep the CDS fixed across all constructs and use "
                "equal weights (0.25 each) so that scoring reflects barcode quality only — not "
                "CDS expression differences."
            )

        # Bar chart: composite score per construct (top 20)
        top20 = df.nlargest(20, "Composite_Score")
        fig = px.bar(
            top20,
            x="Name",
            y="Composite_Score",
            title="Composite Score per Construct (Top 20)",
            color="Composite_Score",
            color_continuous_scale="Reds",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Summary table with score breakdown columns
        display_cols = ["Name", "Barcode", "GC_Pct", "Length", "Score_CAI", "Score_MFE", "Score_GC", "Score_UTR", "Total_Score", "3UTR_Name"]
        # Only include columns that exist in the dataframe
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].rename(
                columns={
                    "GC_Pct": "GC%",
                    "Score_CAI": "Score_CAI",
                    "Score_MFE": "Score_MFE",
                    "Score_GC": "Score_GC",
                    "Score_UTR": "Score_UTR",
                    "Total_Score": "Total_Score",
                    "3UTR_Name": "3'UTR used",
                }
            ),
            use_container_width=True,
        )

    # ── Tab 2: Sequence Viewer ──
    with tab2:
        construct_names = df["Name"].tolist()
        selected_construct = st.selectbox("Select construct", construct_names)

        row = df[df["Name"] == selected_construct].iloc[0]

        # Colored blocks
        st.markdown("#### Annotated Construct")

        blocks_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:2px;align-items:center;font-family:monospace;font-size:12px;">'
            f'<span style="background:#2196F3;color:white;padding:4px 8px;border-radius:4px;">[5\'Cap: {row["Cap"]}]</span>'
            f'<span style="background:#4CAF50;color:white;padding:4px 8px;border-radius:4px;">[5\'UTR: {len(row["5UTR"])}nt]</span>'
            f'<span style="background:#FF9800;color:white;padding:4px 8px;border-radius:4px;">[CDS: {len(row["CDS"])}nt]</span>'
            f'<span style="background:#F44336;color:white;padding:4px 8px;border-radius:4px;">[Barcode: {len(row["Barcode"])}nt]</span>'
            f'<span style="background:#9C27B0;color:white;padding:4px 8px;border-radius:4px;">[3\'UTR: {len(row["3UTR"])}nt]</span>'
            f'<span style="background:#9E9E9E;color:white;padding:4px 8px;border-radius:4px;">[Poly-A({row["PolyA_Length"]})]</span>'
        )
        if st.session_state.get("add_invdc", False):
            blocks_html += '<span style="background:#8B0000;color:white;padding:4px 8px;border-radius:4px;">[invdC]</span>'
        blocks_html += "</div>"

        st.markdown(blocks_html, unsafe_allow_html=True)

        # DNA template
        st.markdown("#### DNA Template Sequence")
        st.code(row["DNA_Template"], language=None)

        # Key stats
        st.markdown("#### Key Stats")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("CAI Score", f"{row['CAI_Score']:.4f}")
        sc2.metric("GC%", f"{row['GC_Pct']:.1f}%")
        sc3.metric("Total Length", f"{row['Length']} nt")
        sc4.metric("Poly-A Length", f"{row['PolyA_Length']} nt")

    # ── Tab 3: DNA Templates ──
    with tab3:
        st.markdown("**Order these as dsDNA templates or gBlocks for IVT**")

        dna_df = df[["Name", "T7_Promoter", "DNA_Template", "Length"]].copy()
        dna_df = dna_df.rename(columns={"DNA_Template": "DNA_Template_Sequence"})
        st.dataframe(dna_df, use_container_width=True)

        # Download FASTA
        fasta_lines = []
        for _, r in df.iterrows():
            fasta_lines.append(f">{r['Name']}_DNA_template")
            fasta_lines.append(r["DNA_Template"])
        fasta_text = "\n".join(fasta_lines)

        st.download_button(
            "📥 Download DNA Templates (FASTA)",
            data=fasta_text,
            file_name="dna_templates.fasta",
            mime="text/plain",
        )

        # Download CSV
        csv_buf = io.StringIO()
        dna_df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Download DNA Templates (CSV)",
            data=csv_buf.getvalue(),
            file_name="dna_templates.csv",
            mime="text/csv",
        )

    # ── Tab 4: invdC Ligation Oligos ──
    with tab4:
        if st.session_state.get("add_invdc", False):
            st.markdown(
                "After IVT, ligate a pre-synthesized invdC oligo to the 3' end of each mRNA "
                "using T4 RNA Ligase 1. Order these oligos from IDT with the /3InvdC/ modification."
            )

            ligation_oligo = st.text_input(
                "Ligation oligo sequence",
                value="TTTTTTTTTT",
                help="10-nt poly-T splint oligo",
            )

            oligo_data = []
            for _, r in df.iterrows():
                idt_seq = ligation_oligo + "/3InvdC/"
                oligo_data.append({
                    "Name": r["Name"],
                    "Ligation_Oligo_Sequence": ligation_oligo,
                    "IDT_Order_Sequence": idt_seq,
                    "Notes": "Order from IDT with 3' inverted dC modification",
                })

            oligo_df = pd.DataFrame(oligo_data)
            st.dataframe(oligo_df, use_container_width=True)

            csv_buf2 = io.StringIO()
            oligo_df.to_csv(csv_buf2, index=False)
            st.download_button(
                "📥 Download IDT Order Sheet (CSV)",
                data=csv_buf2.getvalue(),
                file_name="invdc_oligos_idt_order.csv",
                mime="text/csv",
            )
        else:
            st.info("Enable 'Add invdC at 3' end' in the sidebar to see ligation oligo details.")

    # ── Step 4: Export ──
    st.header("Step 4: Export")

    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        # mRNA Sequences FASTA
        mrna_fasta = []
        for _, r in df.iterrows():
            mrna_fasta.append(f">{r['Name']}")
            mrna_fasta.append(r["mRNA_Sequence"])
        st.download_button(
            "📥 mRNA Sequences (FASTA)",
            data="\n".join(mrna_fasta),
            file_name="mrna_sequences.fasta",
            mime="text/plain",
            use_container_width=True,
        )

    with exp2:
        # Full Library Metadata CSV
        csv_full = io.StringIO()
        df.to_csv(csv_full, index=False)
        st.download_button(
            "📥 Full Library Metadata (CSV)",
            data=csv_full.getvalue(),
            file_name="mrna_library_metadata.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp3:
        # Synthesis Order Sheet
        order_data = []
        for _, r in df.iterrows():
            row_data = {
                "Name": r["Name"],
                "DNA_Template": r["DNA_Template"],
                "Template_Length": len(r["DNA_Template"]),
            }
            if st.session_state.get("add_invdc", False):
                row_data["invdC_Oligo"] = "TTTTTTTTTT/3InvdC/"
            order_data.append(row_data)
        order_df = pd.DataFrame(order_data)
        csv_order = io.StringIO()
        order_df.to_csv(csv_order, index=False)
        st.download_button(
            "📥 Synthesis Order Sheet (CSV)",
            data=csv_order.getvalue(),
            file_name="synthesis_order_sheet.csv",
            mime="text/csv",
            use_container_width=True,
        )
