"""
Tests for mrna_design modules.
Run with:  pytest tests/ -v
"""
import pytest
from pathlib import Path
import tempfile, os

from mrna_design.assembler import (
    read_fasta, assemble_construct, assemble_library, KOZAK_SEQUENCES
)
from mrna_design.scorer import (
    calc_gc_content, gc_score, cai_score, mfe_stability_score,
    utr_accessibility_score, codon_pair_bias_score, uridine_depletion_score,
    cpg_depletion_score, score_library
)
from mrna_design.barcode import (
    hamming_distance, gc_fraction, has_homopolymer,
    generate_barcode_pool, assign_barcodes, encode_peptide_barcode
)
from mrna_design.optimizer import translate, optimize_codon_usage
from mrna_design.qc import (
    check_cpg_density, check_uridine_content, check_homopolymer_runs,
    check_local_gc_extremes, check_restriction_sites,
    check_premature_polya_signals, check_internal_aug_in_utr,
    check_inverted_repeats, check_mirna_seed_matches,
    run_qc, qc_library,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

FASTA_CONTENT_UTR5 = ">UTR5_A\nGCCACCAUGG\n>UTR5_B\nAAAAAAAAAAAA\n"
FASTA_CONTENT_ORF  = ">ORF_A\nAUGGAGAAAGACGCC\n"
FASTA_CONTENT_UTR3 = ">UTR3_A\nUGCUAGUGAAUUUCG\n"

@pytest.fixture
def tmp_fasta_dirs(tmp_path):
    """Create temporary directories with FASTA files for each region."""
    for dirname, content in [
        ("utr5", FASTA_CONTENT_UTR5),
        ("orf",  FASTA_CONTENT_ORF),
        ("utr3", FASTA_CONTENT_UTR3),
    ]:
        d = tmp_path / dirname
        d.mkdir()
        (d / "seqs.fasta").write_text(content)
    return tmp_path


# ── assembler ─────────────────────────────────────────────────────────────────

def test_read_fasta_basic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(FASTA_CONTENT_UTR5)
        f.flush()
        records = read_fasta(Path(f.name))
    os.unlink(f.name)
    assert len(records) == 2
    assert records[0]["name"] == "UTR5_A"
    assert records[0]["seq"]  == "GCCACCAUGG"


def test_read_fasta_t_to_u():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">test\nATGGAGAAAGACGCC\n")
        f.flush()
        records = read_fasta(Path(f.name))
    os.unlink(f.name)
    assert "T" not in records[0]["seq"]
    assert "U" in records[0]["seq"]


def test_assemble_construct_keys():
    u5  = {"name": "u5", "seq": "AAAAAA"}
    orf = {"name": "orf", "seq": "AUGCCC"}
    u3  = {"name": "u3", "seq": "UUUUUU"}
    result = assemble_construct(u5, orf, u3)
    assert "full_sequence" in result
    assert "total_length"  in result
    assert result["utr5_name"] == "u5"


def test_assemble_construct_polya():
    u5  = {"name": "u5", "seq": "A"}
    orf = {"name": "orf", "seq": "AUG"}
    u3  = {"name": "u3", "seq": "U"}
    result = assemble_construct(u5, orf, u3, polya_len=50)
    assert result["full_sequence"].endswith("A" * 50)


def test_assemble_library_count(tmp_fasta_dirs):
    lib = assemble_library(
        utr5_dir=tmp_fasta_dirs / "utr5",
        orf_dir =tmp_fasta_dirs / "orf",
        utr3_dir=tmp_fasta_dirs / "utr3",
    )
    # 2 UTR5 × 1 ORF × 1 UTR3 = 2
    assert len(lib) == 2


def test_assemble_library_ids(tmp_fasta_dirs):
    lib = assemble_library(
        utr5_dir=tmp_fasta_dirs / "utr5",
        orf_dir =tmp_fasta_dirs / "orf",
        utr3_dir=tmp_fasta_dirs / "utr3",
    )
    ids = [c["id"] for c in lib]
    assert len(set(ids)) == len(ids)  # all unique


# ── scorer ────────────────────────────────────────────────────────────────────

def test_gc_content_pure_gc():
    assert calc_gc_content("GCGCGCGC") == 1.0

def test_gc_content_pure_au():
    assert calc_gc_content("AUAUAUAU") == 0.0

def test_gc_score_optimal():
    # ~52% GC should score near 1.0
    seq = "GCGCGCGCAUAUAUAU"  # 50% GC
    assert gc_score(seq) > 0.8

def test_cai_score_range():
    seq = "AUGCUCGAGCUGGAG" * 5
    assert 0.0 <= cai_score(seq) <= 1.0

def test_mfe_score_range():
    seq = "AUGCUCGAGCUGGAG" * 10
    assert 0.0 <= mfe_stability_score(seq) <= 1.0

def test_score_library_composite(tmp_fasta_dirs):
    from mrna_design.assembler import assemble_library
    lib = assemble_library(
        utr5_dir=tmp_fasta_dirs / "utr5",
        orf_dir =tmp_fasta_dirs / "orf",
        utr3_dir=tmp_fasta_dirs / "utr3",
    )
    scored = score_library(lib)
    for c in scored:
        assert "composite_score" in c
        assert 0.0 <= c["composite_score"] <= 1.0


# ── barcode ───────────────────────────────────────────────────────────────────

def test_hamming_identical():
    assert hamming_distance("AAAA", "AAAA") == 0

def test_hamming_all_diff():
    assert hamming_distance("AAAA", "CCCC") == 4

def test_gc_fraction():
    assert gc_fraction("GCAU") == 0.5

def test_has_homopolymer():
    assert has_homopolymer("AAAAACC", run_len=4)
    assert not has_homopolymer("ACGUACGU", run_len=4)

def test_generate_barcode_pool_unique():
    pool = generate_barcode_pool(n=20, length=16, min_hamming=3)
    assert len(pool) == 20
    assert len(set(pool)) == 20

def test_generate_barcode_pool_hamming():
    pool = generate_barcode_pool(n=10, length=16, min_hamming=3)
    for i, a in enumerate(pool):
        for b in pool[i+1:]:
            assert hamming_distance(a, b) >= 3

def test_assign_barcodes_count():
    dummy_library = [{"id": f"X{i}", "full_sequence": "AUG"} for i in range(5)]
    result = assign_barcodes(dummy_library)
    assert all("barcode" in c for c in result)
    barcodes = [c["barcode"] for c in result]
    assert len(set(barcodes)) == 5

def test_peptide_barcode_encodes():
    seq = "AUGCUGAAG"  # M L K
    peptide = encode_peptide_barcode(seq)
    assert peptide == "MLK"


# ── optimizer ─────────────────────────────────────────────────────────────────

def test_translate():
    assert translate("AUGCUG") == "ML"

def test_optimize_preserves_aa():
    seq = "AUGCUGAAGCCCGAG"
    opt = optimize_codon_usage(seq, method="max_cai")
    assert translate(seq) == translate(opt)

def test_optimize_weighted_preserves_aa():
    seq = "AUGCUGAAGCCCGAG"
    opt = optimize_codon_usage(seq, method="weighted")
    assert translate(seq) == translate(opt)


def test_optimize_balanced_preserves_aa():
    seq = "AUGCUGAAGCCCGAG"
    opt = optimize_codon_usage(seq, method="balanced")
    assert translate(seq) == translate(opt)


# ── scorer (advanced) ─────────────────────────────────────────────────────────

def test_cai_score_geometric_mean():
    """CAI should use geometric mean — all optimal codons → score near 1.0."""
    # All high-frequency codons
    seq = "UUCCUGAUCGUGUGG"  # F(UUC) L(CUG) I(AUC) V(GUG) W(UGG)
    score = cai_score(seq)
    assert score > 0.8

def test_mfe_dinucleotide_model():
    """GC-rich sequences should score higher on MFE (more stable)."""
    gc_rich = "GCGCGCGCGCGCGCGCGCGC"
    au_rich = "AUAUAUAUAUAUAUAUAUAU"
    assert mfe_stability_score(gc_rich) > mfe_stability_score(au_rich)

def test_codon_pair_bias_range():
    seq = "AUGCUGAAGCCCGAG" * 5
    score = codon_pair_bias_score(seq)
    assert 0.0 <= score <= 1.0

def test_uridine_depletion_optimal():
    """Sequence with ~17% U should score highest."""
    # Build sequence with approximately 17% U content
    seq = "GCCACCGUGCCACCGU" * 3  # ~12.5% U — still good range
    score = uridine_depletion_score(seq)
    # Sequence with 25% U should score lower
    high_u_seq = "AUGCUUUUUGAGCCC" * 3  # ~33% U
    high_u_score = uridine_depletion_score(high_u_seq)
    assert score > high_u_score

def test_cpg_depletion_low_cpg():
    """Sequence with no CpG should score 1.0."""
    seq = "AAUAUUAUAAUUAAU"
    assert cpg_depletion_score(seq) == 1.0

def test_cpg_depletion_high_cpg():
    """Sequence with many CpG should score low."""
    seq = "CGCGCGCGCGCGCGCG"
    assert cpg_depletion_score(seq) < 0.3


# ── QC module ─────────────────────────────────────────────────────────────────

def test_qc_cpg_density_pass():
    # Low CpG sequence
    result = check_cpg_density("AAUAUUAUAAUUAAUGGCCAA")
    assert result.passed

def test_qc_cpg_density_fail():
    # High CpG sequence
    result = check_cpg_density("CGCGCGCGCGCGCGCGCGCG", threshold=0.02)
    assert not result.passed

def test_qc_homopolymer_pass():
    result = check_homopolymer_runs("AUGCUGAAGCCC", max_run=5)
    assert result.passed

def test_qc_homopolymer_fail():
    result = check_homopolymer_runs("AUGAAAAAAGCCC", max_run=5)
    assert not result.passed

def test_qc_local_gc_extremes():
    # Very high GC window
    seq = "G" * 50 + "AUAU" * 25
    result = check_local_gc_extremes(seq, window=50, gc_max=0.80)
    assert not result.passed

def test_qc_restriction_sites():
    # Contains EcoRI site (GAAUUC)
    seq = "AUGCCCGAAUUCAAAGGG"
    result = check_restriction_sites(seq)
    assert not result.passed

def test_qc_premature_polya():
    # Contains AAUAAA in middle
    seq = "AUGCCC" + "AAUAAA" + "GGGCCC" * 20
    result = check_premature_polya_signals(seq)
    assert not result.passed

def test_qc_internal_aug():
    result = check_internal_aug_in_utr("GCCAUGCCCAUGAAA")
    assert not result.passed
    assert len(result.positions) == 2

def test_qc_no_internal_aug():
    result = check_internal_aug_in_utr("GCCACCCCCAAA")
    assert result.passed

def test_qc_mirna_seeds():
    # Should not crash on short sequences
    result = check_mirna_seed_matches("AUGCCC")
    assert result.passed

def test_qc_library_integration(tmp_fasta_dirs):
    """Full QC pipeline should add qc fields to constructs."""
    from mrna_design.assembler import assemble_library
    from mrna_design.scorer import score_library
    lib = assemble_library(
        utr5_dir=tmp_fasta_dirs / "utr5",
        orf_dir=tmp_fasta_dirs / "orf",
        utr3_dir=tmp_fasta_dirs / "utr3",
    )
    scored = score_library(lib)
    qc_result = qc_library(scored)
    for c in qc_result:
        assert "qc_passed" in c
        assert "qc_warnings" in c
        assert "qc_penalty" in c
