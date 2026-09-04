# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Configuration schema for the optional LIT workflow."""

from dataclasses import field

from jaqmc.utils.config import configurable_dataclass
from jaqmc.wavefunction.output.envelope import EnvelopeType


@configurable_dataclass
class FrequencyGridConfig:
    """Reported spectrum grid."""

    minimum: float = 0.0
    maximum: float = 1.0
    points: int = 501
    values: tuple[float, ...] = field(default_factory=tuple)


@configurable_dataclass
class GroundStateConfig:
    """Ground-state checkpoint and sampling settings."""

    checkpoint_path: str = ""
    allow_untrained: bool = False
    energy: float | None = None
    energy_steps: int = 2
    burn_in: int = 20


@configurable_dataclass
class SourceConfig:
    """Dipole-source construction, caching, and distillation settings."""

    center_steps: int = 4
    center_override: float | tuple[float, float, float] | None = None
    norm_override: float | tuple[float, float, float] | None = None
    burn_in: int = 20
    floor: float = 1e-4
    train_pool_batches: int = 32
    eval_pool_batches: int = 8
    pool_stride: int = 1
    pool_dir: str = ""
    reuse_pool: bool = True
    save_pool: bool = True
    distillation_iterations: int = 1000


@configurable_dataclass
class ParallelConfig:
    """Batching and process-local device parallelism."""

    mode: str = "off"
    train_batch_size: int = 0
    eval_batch_size: int = 0
    train_batch_size_per_device: int = 0
    eval_batch_size_per_device: int = 0


@configurable_dataclass
class SolverConfig:
    """Response-state optimizer and held-out selection settings."""

    iterations: int = 200
    learning_rate: float = 1e-3
    reverse_kl_weight: float = 1.0
    spring_epsilon: float = 1e-3
    spring_decay: float = 0.99
    spring_damping_floor: float = 1e-12
    sr_max_norm: float | None = 0.1
    sr_score_epsilon: float = 1e-10
    warm_start_omega: float | None = -3.674932217565499
    warm_start_iterations: int = 100
    plateau_start_iteration: int = 0
    plateau_patience_iterations: int = 0
    plateau_min_delta: float = 1e-5
    selection_interval: int = 50
    log_interval: int = 50


@configurable_dataclass
class ContinuationConfig:
    """Adaptive frequency continuation and recovery settings."""

    iterations: int = 100
    step_fraction: float = 0.2
    step_growth_factor: float = 1.25
    fidelity_retention: float = 0.95
    ess_fraction_minimum: float = 0.0
    allow_minimum_step_recovery: bool = True
    minimum_step: float | None = None
    maximum_points: int = 256
    restore_path: str = ""


@configurable_dataclass
class AnsatzConfig:
    """Independent response-wavefunction architecture and sector guards."""

    determinants: int = 16
    hidden_dims_single: tuple[int, ...] = field(
        default_factory=lambda: (256, 256, 256, 256)
    )
    hidden_dims_double: tuple[int, ...] = field(
        default_factory=lambda: (32, 32, 32, 32)
    )
    use_last_layer: bool = False
    envelope: EnvelopeType = EnvelopeType.abs_isotropic
    orbitals_spin_split: bool = True
    parity_eval_batch_size: int = 256
    sector_tolerance: float = 1e-5
    atomic_source_parity_max_loss: float = 1e-3
    atomic_ground_parity_max_loss: float = 1e-3


@configurable_dataclass
class LITConfig:
    """Configuration for a molecular electric-dipole LIT spectrum."""

    eta: float = 0.02
    axes: str = "xyz"
    output_filename: str = "lit_spectrum.npz"
    omega: FrequencyGridConfig = field(default_factory=FrequencyGridConfig)
    ground: GroundStateConfig = field(default_factory=GroundStateConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    ansatz: AnsatzConfig = field(default_factory=AnsatzConfig)
