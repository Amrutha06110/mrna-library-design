"""
config_model.py — Pydantic-based configuration validation for mRNA Library Design.

Validates config.yaml structure, field types, ranges, and score-weight normalization.
Fails fast with actionable error messages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ScoringWeights(BaseModel):
    """Scoring weights for multi-objective mRNA scoring. Must sum to ~1.0."""

    cai: float = Field(0.20, ge=0.0, le=1.0, description="Codon Adaptation Index weight")
    mfe_stability: float = Field(0.15, ge=0.0, le=1.0, description="MFE stability weight")
    gc_content: float = Field(0.10, ge=0.0, le=1.0, description="GC content optimality weight")
    utr_access: float = Field(0.12, ge=0.0, le=1.0, description="UTR accessibility weight")
    codon_pair: float = Field(0.10, ge=0.0, le=1.0, description="Codon pair bias weight")
    u_depletion: float = Field(0.08, ge=0.0, le=1.0, description="Uridine depletion weight")
    cpg_depletion: float = Field(0.05, ge=0.0, le=1.0, description="CpG depletion weight")
    translation_eff: float = Field(0.10, ge=0.0, le=1.0, description="Translation efficiency weight")
    codon_ramp: float = Field(0.05, ge=0.0, le=1.0, description="Codon ramp weight")
    codon_diversity: float = Field(0.05, ge=0.0, le=1.0, description="Codon diversity weight")

    @model_validator(mode="after")
    def normalize_weights(self) -> "ScoringWeights":
        """Auto-normalize weights to sum to 1.0 if they don't already."""
        total = (
            self.cai + self.mfe_stability + self.gc_content + self.utr_access
            + self.codon_pair + self.u_depletion + self.cpg_depletion
            + self.translation_eff + self.codon_ramp + self.codon_diversity
        )
        if total == 0:
            raise ValueError("All scoring weights are zero; at least one must be positive.")
        if abs(total - 1.0) > 0.01:
            # Auto-normalize
            self.cai /= total
            self.mfe_stability /= total
            self.gc_content /= total
            self.utr_access /= total
            self.codon_pair /= total
            self.u_depletion /= total
            self.cpg_depletion /= total
            self.translation_eff /= total
            self.codon_ramp /= total
            self.codon_diversity /= total
        return self

    def to_dict(self) -> dict[str, float]:
        """Return weights as a plain dict for backward compatibility."""
        return self.model_dump()


class BarcodingConfig(BaseModel):
    """Barcoding settings."""

    mode: Literal["dna", "peptide", "quart"] = "dna"
    length: int = Field(16, ge=4, le=64, description="Barcode length in nucleotides")
    min_hamming: int = Field(3, ge=1, le=16, description="Min pairwise Hamming distance")
    gc_min: float = Field(0.40, ge=0.0, le=1.0)
    gc_max: float = Field(0.60, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_gc_range(self) -> "BarcodingConfig":
        if self.gc_min >= self.gc_max:
            raise ValueError(
                f"barcoding.gc_min ({self.gc_min}) must be less than gc_max ({self.gc_max})"
            )
        return self


class OptimizerConfig(BaseModel):
    """Codon optimizer settings."""

    method: Literal["max_cai", "weighted", "balanced"] = "balanced"
    tool: Literal["builtin", "vaxpress", "lineardesign"] = "builtin"


class QCConfig(BaseModel):
    """Quality control thresholds."""

    cpg_max_density: float = Field(0.02, ge=0.0, le=1.0)
    uridine_max_fraction: float = Field(0.25, ge=0.0, le=1.0)
    max_homopolymer_run: int = Field(5, ge=2, le=20)
    gc_window: int = Field(50, ge=10, le=500)
    local_gc_min: float = Field(0.25, ge=0.0, le=1.0)
    local_gc_max: float = Field(0.80, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_gc_range(self) -> "QCConfig":
        if self.local_gc_min >= self.local_gc_max:
            raise ValueError(
                f"qc.local_gc_min ({self.local_gc_min}) must be less than local_gc_max ({self.local_gc_max})"
            )
        return self


class FiltersConfig(BaseModel):
    """Post-scoring filter thresholds."""

    min_composite_score: float = Field(0.0, ge=0.0, le=1.0)
    gc_min: float = Field(0.40, ge=0.0, le=1.0)
    gc_max: float = Field(0.70, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_gc_range(self) -> "FiltersConfig":
        if self.gc_min >= self.gc_max:
            raise ValueError(
                f"filters.gc_min ({self.gc_min}) must be less than gc_max ({self.gc_max})"
            )
        return self


class PipelineConfig(BaseModel):
    """Top-level mRNA Library Design configuration."""

    utr5_dir: str = "data/utr5"
    orf_dir: str = "data/orf"
    utr3_dir: str = "data/utr3"
    output_dir: str = "outputs"

    cap: Literal["m7G", "cap1", "arca", "none"] = "m7G"
    kozak: Literal["strong", "moderate", "none"] = "strong"
    polya_length: int = Field(100, ge=0, le=500)
    max_combinations: int | None = Field(1000, ge=1)

    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    barcoding: BarcodingConfig = Field(default_factory=BarcodingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)

    # Runtime options (set via CLI, not usually in config file)
    chunk_size: int = Field(5000, ge=100, le=1_000_000, description="Chunk size for streaming assembly/scoring")
    workers: int = Field(1, ge=1, le=32, description="Number of parallel scoring workers")

    @field_validator("max_combinations", mode="before")
    @classmethod
    def allow_null_max_combinations(cls, v):
        """Allow null/None in config to mean 'no cap'."""
        if v is None:
            return None
        return v


def load_and_validate_config(config_path: Path | None = None, overrides: dict | None = None) -> PipelineConfig:
    """
    Load YAML config, apply overrides, validate, and return a PipelineConfig.

    Args:
        config_path: Path to config YAML file (uses defaults if None or missing).
        overrides: Dict of CLI override values (None values are skipped).

    Returns:
        Validated PipelineConfig instance.

    Raises:
        ValueError: If config validation fails (with actionable error message).
    """
    import yaml

    raw: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    # Apply overrides (only non-None values)
    if overrides:
        for key, val in overrides.items():
            if val is not None:
                raw[key] = val

    try:
        return PipelineConfig(**raw)
    except Exception as e:
        raise ValueError(f"Configuration validation failed:\n  {e}") from e
