from enum import IntEnum
from typing import NamedTuple

from ._teaserpp import (
     OMP_MAX_THREADS,
     CertificationResult,
     DRSCertifier,
     EigSolverType,
     InlierGraphFormulation,
     InlierSelectionMode,
     RegistrationSolution,
     RobustRegistrationSolver,
     RotationEstimationAlgorithm,
)


class RobustRegistrationSolverParams(NamedTuple):
     noise_bound: float = 0.01
     cbar2: float = 1
     estimate_scaling: bool = True
     rotation_estimation_algorithm: RotationEstimationAlgorithm = RotationEstimationAlgorithm.GNC_TLS
     rotation_gnc_factor: float = 1.4
     rotation_max_iterations: int = 100
     rotation_cost_threshold: float = 1e-6
     rotation_tim_graph: InlierGraphFormulation = InlierGraphFormulation.CHAIN
     inlier_selection_mode: InlierSelectionMode = InlierSelectionMode.PMC_EXACT
     kcore_heuristic_threshold: float = 0.5
     use_max_clique: bool = True
     max_clique_exact_solution: bool = True
     max_clique_time_limit: int = 3000
     max_clique_num_threads: int = OMP_MAX_THREADS
