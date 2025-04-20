from functools import wraps
from typing import Callable, NamedTuple

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

# Backwards compatibility with v1.0 so we don't break code
RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM = RotationEstimationAlgorithm
RobustRegistrationSolver.INLIER_SELECTION_MODE = InlierSelectionMode
RobustRegistrationSolver.INLIER_GRAPH_FORMULATION = InlierGraphFormulation
DRSCertifier.EIG_SOLVER_TYPE = EigSolverType


class RobustRegistrationSolverParams(NamedTuple):
    noise_bound: float = 0.01
    cbar2: float = 1
    estimate_scaling: bool = True
    rotation_estimation_algorithm: RotationEstimationAlgorithm = (
        RotationEstimationAlgorithm.GNC_TLS
    )
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


# Do some Python magic
def __init_deco(f: Callable[..., None]):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        f(self, *args, **kwargs)
        self._params = args

    return wrapper


@property
def __params_getter(self) -> RobustRegistrationSolverParams:
    return self._params


RobustRegistrationSolver.__init__ = __init_deco(RobustRegistrationSolver.__init__)
setattr(RobustRegistrationSolver, "params", __params_getter)
