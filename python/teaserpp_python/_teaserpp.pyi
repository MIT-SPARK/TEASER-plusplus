from enum import Enum
from typing import List, overload

import numpy as np
from typing_extensions import Final

from . import RobustRegistrationSolverParams

OMP_MAX_THREADS: Final[int]

class RegistrationSolution:
    scale: float
    translation: np.ndarray
    rotation: np.ndarray

    def __repr__(self) -> str: ...

class RotationEstimationAlgorithm(Enum):
    """
    An enum class representing the available GNC rotation estimation
    algorithms.
    """

    #: see H. Yang, P. Antonante, V. Tzoumas, and L. Carlone,
    #: “Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal
    #: Solvers to Global Outlier Rejection,” arXiv:1909.08605 [cs, math],
    #: Sep. 2019.
    GNC_TLS = 0

    #: see Q.-Y. Zhou, J. Park, and V. Koltun, “Fast Global Registration,”
    #: in Computer Vision – ECCV 2016, Cham, 2016, vol. 9906, pp. 766–782. and
    #: H. Yang, P. Antonante, V. Tzoumas, and L. Carlone, “Graduated
    #: Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to
    #: Global Outlier Rejection,” arXiv:1909.08605 [cs, math], Sep. 2019.
    FGR = 1

    #: see H. Lim et al., "A Single Correspondence Is Enough: Robust Global
    #: Registration to Avoid Degeneracy in Urban Environments," in Robotics -
    #: ICRA 2022, pp. 8010-8017 arXiv:2203.06612 [cs], Mar. 2022.
    QUARTO = 2

class InlierGraphFormulation(Enum):
    """
    Enum representing the formulation of the TIM graph after maximum clique
    filtering.
    """

    #: formulate TIMs by only calculating the TIMs for consecutive measurements
    CHAIN = 0

    #: formulate a fully connected TIM graph
    COMPLETE = 1

class InlierSelectionMode(Enum):
    """
    Enum representing the type of graph-based inlier selection algorithm to
    use.
    """

    #: Use PMC to find exact clique from the inlier graph
    PMC_EXACT = 0

    #: Use PMC's heuristic finder to find approximate max clique
    PMC_HEU = 1

    #: Use k-core heuristic to select inliers
    KCORE_HEU = 2

    #: No inlier selection
    NONE = 3

class EigSolverType(Enum):
    EIGEN = 0
    SPECTRA = 1

class CertificationResult:
    is_optimal: bool
    best_suboptimality: float
    suboptimality_traj: List[float]

    def __repr__(self) -> str: ...

class RobustRegistrationSolver:
    """
    Solve registration problems robustly.

    For more information, please refer to:
    H. Yang, J. Shi, and L. Carlone, “TEASER: Fast and Certifiable Point Cloud
    Registration,” arXiv:2001.07715 [cs, math], Jan. 2020.
    """

    ROTATION_ESTIMATION_ALGORITHM = RotationEstimationAlgorithm
    INLIER_SELECTION_MODE = InlierSelectionMode
    INLIER_GRAPH_FORMULATION = InlierGraphFormulation

    class Params:
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

    @overload
    def __init__(self, params: Params): ...

    @overload
    def __init__(
        self,
        noise_bound: float = 0.01,
        cbar2: float = 1,
        estimate_scaling: bool = True,
        rotation_estimation_algorithm: RotationEstimationAlgorithm = (
            RotationEstimationAlgorithm.GNC_TLS
        ),
        rotation_gnc_factor: float = 1.4,
        rotation_max_iterations: int = 100,
        rotation_cost_threshold: float = 1e-6,
        rotation_tim_graph: InlierGraphFormulation = InlierGraphFormulation.CHAIN,
        inlier_selection_mode: InlierSelectionMode = InlierSelectionMode.PMC_EXACT,
        kcore_heuristic_threshold: float = 0.5,
        use_max_clique: bool = True,
        max_clique_exact_solution: bool = True,
        max_clique_time_limit: int = 3000,
        max_clique_num_threads: int = OMP_MAX_THREADS,
    ) -> None:
        """
        Parameters
        ----------
        noise_bound : float
            A bound on the noise of each provided measurement.

        cbar2 : float
            Square of the ratio between acceptable noise and noise bound.
            Default value of 1 is typical.

        estimate_scaling : bool
            Whether the scale is known. If set to False, the solver assumes no
            scale differences between the src and dst points. If set to True,
            the solver will first solve for scale. When the solver does not
            estimate scale, it solves the registration problem much faster.

        rotation_estimation_algorithm : RotationEstimationAlgorithm
            The algorithm to use to estimate rotations.

        rotation_gnc_factor : float
            Factor to multiple/divide the control parameter in the GNC
            algorithm.
            For FGR: the algorithm divides the control parameter by the factor
            every iteration.
            For GNC-TLS: the algorithm multiples the control parameter by the
            factor every iteration.

        rotation_cost_threshold : float
            Cost threshold for the GNC rotation estimators.
            For FGR / GNC-TLS algorithm, the cost thresholds represent
            different meanings.
            For FGR: the cost threshold compares with the computed cost at each
            iteration.
            For GNC-TLS: the cost threshold compares with the difference
            between costs of consecutive iterations.

        rotation_tim_graph : InlierGraphFormulation
            Type of TIM graph given to GNC rotation solver.

        kcore_heuristic_threshold : InlierSelectionMode
            The threshold ratio for determining whether to skip max
            clique and go straightly to GNC rotation estimation. Set this to 1
            to always use exact max clique selection, 0 to always skip exact max
            clique selection.

        use_max_clique : bool
            Deprecated. Use inlier_selection_mode instead
            Set this to true to enable max clique inlier selection,
            False to skip it.

        max_clique_exact_solution : bool
            Deprecated. Use inlier_selection_mode instead
            Set this to false to enable heuristic only max clique finding.

        max_clique_time_limit : int
            Time limit on running the max clique algorithm (in seconds).

        max_clique_num_threads : Optional[int]
            Number of threads to use to run the max clique algorithm.
            Defaults to using the maximum number.

        Note
        ----
        Note that the use_max_clique option takes precedence. In other words,
        if use_max_clique is set to false, then kcore_heuristic_threshold
        will be ignored. If use_max_clique is set to true, then the
        following will happen: if the max core number of the inlier graph is
        lower than the kcore_heuristic_threshold as a percentage of the
        total nodes in the inlier graph, then the code will proceed to call
        the max clique finder. Otherwise, the graph will be directly fed to
        the GNC rotation solver.
        """
        ...

    @property
    def params(self) -> RobustRegistrationSolverParams: ...
    def getParams(self) -> Params: ...
    def reset(self) -> None: ...
    def solve(self, src: np.ndarray, dst: np.ndarray) -> None:
        """
        Solve for scale, translation and rotation from src to dst.
        """
        ...

    @property
    def solution(self) -> RegistrationSolution: ...
    def getSolution(self) -> RegistrationSolution: ...
    @property
    def gnc_rotation_cost_at_termination(self): ...
    def getGNCRotationCostAtTermination(self) -> float:
        """
        Return the cost at termination of the GNC rotation solver. Can be used
        to assess the quality of the solution.
        """
        ...

    @property
    def scale_inliers_mask(self) -> np.ndarray:
        """
        A boolean ndarray indicating whether specific measurements are
        scale inliers according to scales.
        """
        ...

    def getScaleInliersMask(self) -> np.ndarray:
        """
        Return a boolean ndarray indicating whether specific measurements are
        scale inliers according to scales.
        """
        ...

    @property
    def scale_inliers_map(self) -> np.ndarray:
        """
        The index map for scale inliers (equivalent to the index map for
        TIMs).
        """
        ...

    def getScaleInliersMap(self) -> np.ndarray:
        """
        Return the index map for scale inliers (equivalent to the index map for
        TIMs).

        Returns
        -------
        A 2-by-(number of TIMs) ndarray. Entries in one column represent
        the indices of the two measurements used to calculate the corresponding
        TIM.
        """
        ...

    @property
    def scale_inliers(self) -> List[int]: ...
    def getScaleInliers(self) -> List[int]:
        """
        Return inlier TIMs from scale estimation

        Returns
        -------
        A vector of tuples. Entries in each tuple represent the indices of
        the two measurements used to calculate the corresponding TIM:
        measurement at indice 0 minus measurement at indice 1.
        """
        ...

    @property
    def rotation_inliers_mask(self) -> np.ndarray: ...
    def getRotationInliersMask(self) -> np.ndarray:
        """
        Return a boolean ndarray indicating whether specific
        measurements are inliers according to the rotation solver.


        Returns
        -------
        A 1-by-(size of TIMs used in rotation estimation) boolean ndarray.
        It is equivalent to a binary mask on the TIMs used in rotation
        estimation, with true representing that the measurement is an inlier
        after rotation estimation.
        """
        ...

    # TODO: This is obsolete now. Remove or update
    def getRotationInliersMap(self) -> np.ndarray:
        """
        Return the index map for translation inliers (equivalent to max clique).

        Returns
        -------
        A 1-by-(size of max clique) ndarray. Entries represent the indices of
        the original measurements.
        """
        ...

    @property
    def rotation_inliers(self) -> List[int]:
        """
        The index map for translation inliers (equivalent to max clique).
        """
        ...

    def getRotationInliers(self) -> np.ndarray:
        """
        Return a boolean ndarray indicating whether specific
        measurements are inliers according to translation measurements.

        Returns
        -------
        A 1-by-(size of max clique) boolean ndarray. It is equivalent to a
        binary mask on the inlier max clique, with true representing that the
        measurement is an inlier after translation estimation.
        """
        ...

    @property
    def translation_inliers_mask(self) -> np.ndarray:
        """
        A boolean ndarray indicating whether specific
        measurements are inliers according to translation measurements.

        """
        ...

    def getTranslationInliersMask(self) -> np.ndarray:
        """
        Return the index map for translation inliers (equivalent to max clique).

        Returns
        -------
        A 1-by-(size of max clique) ndarray. Entries represent the indices of
        the original measurements.
        """
        ...

    @property
    def translation_inliers_map(self) -> np.ndarray:
        """
        The index map for translation inliers (equivalent to max clique).
        """
        ...

    def getTranslationInliersMap(self) -> np.ndarray:
        """
        Return the index map for translation inliers (equivalent to max clique).

        Returns
        -------
        A 1-by-(size of max clique) ndarray. Entries represent the indices of
        the  original measurements.
        """
        ...

    @property
    def translation_inliers(self) -> List[int]:
        """
        Inliers from translation estimation
        """

        ...

    def getTranslationInliers(self) -> List[int]:
        """
        Return inliers from translation estimation

        Returns
        -------
        A vector of indices of measurements deemed as inliers by translation
        estimation.
        """
        ...

    @property
    def inlier_max_clique(self) -> np.ndarray:
        """
        A boolean ndarray indicating whether specific measurements are
        inliers according to translation measurements.
        """
        ...

    def getInlierMaxClique(self) -> np.ndarray:
        """
        Return a boolean ndarray indicating whether specific measurements are
        inliers according to translation measurements.
        """
        ...

    @property
    def inlier_graph(self) -> np.ndarray: ...
    def getInlierGraph(self) -> np.ndarray: ...
    @property
    def src_tims_map(self) -> np.ndarray:
        """
        The index map of the TIMs built from the source point cloud.
        """
        ...

    def getSrcTIMsMap(self) -> np.ndarray:
        """
        Get the index map of the TIMs built from the source point cloud.
        """
        ...

    @property
    def dst_tims_map(self) -> np.ndarray:
        """
        The index map of the TIMs built from the target point cloud.
        """
        ...

    def getDstTIMsMap(self) -> np.ndarray:
        """
        Get the index map of the TIMs built from the target point cloud.
        """
        ...

    @property
    def src_tims_map_for_rotation(self) -> np.ndarray:
        """
        The index map of the TIMs used in rotation estimation from
        the source point cloud.
        """
        ...

    def getSrcTIMsMapForRotation(self) -> np.ndarray:
        """
        Get the index map of the TIMs used in rotation estimation from
        the source point cloud.
        """
        ...

    @property
    def dst_tims_map_for_rotation(self) -> np.ndarray:
        """
        The index map of the TIMs used in rotation estimation from
        the target point cloud.
        """
        ...

    def getDstTIMsMapForRotation(self) -> np.ndarray:
        """
        Get the index map of the TIMs used in rotation estimation from the
        target point cloud.
        """
        ...

    @property
    def max_clique_src_tims(self) -> np.ndarray:
        """
        Source TIMs built after max clique pruning.
        """
        ...

    def getMaxCliqueSrcTIMs(self) -> List[int]:
        """
        Get src TIMs built after max clique pruning.
        """
        ...

    @property
    def max_clique_dst_tims(self) -> np.ndarray:
        """
        Target TIMs built after max clique pruning.
        """
        ...

    def getMaxCliqueDstTIMs(self) -> List[int]:
        """
        Get dst TIMs built after max clique pruning.
        """
        ...

    @property
    def src_tims(self) -> np.ndarray:
        """
        TIMs built from source point cloud.
        """
        ...

    def getSrcTIMs(self) -> np.ndarray:
        """
        Get TIMs built from source point cloud.
        """
        ...

    @property
    def dst_tims(self) -> np.ndarray:
        """
        TIMs built from source target point cloud.
        """
        ...

    def getDstTIMs(self) -> np.ndarray:
        """
        Get TIMs built from target point cloud.
        """
        ...

class DRSCertifier:
    EIG_SOLVER_TYPE = EigSolverType

    class Params:
        noise_bound: float
        cbar2: float
        sub_optimality: float
        max_iterations: int
        gamma_tau: float
        eig_decomposition_solver: EigSolverType

        def __init__(self) -> None: ...

    def __init__(self, params: Params) -> None: ...
    def certify(
        self, rotation: np.ndarray, src: np.ndarray, dst: np.ndarray, mask: np.ndarray
    ) -> CertificationResult: ...
