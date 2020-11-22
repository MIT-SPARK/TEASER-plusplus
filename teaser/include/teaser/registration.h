/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <memory>
#include <vector>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Geometry>

#include "teaser/graph.h"
#include "teaser/geometry.h"

// TODO: might be a good idea to template Eigen::Vector3f and Eigen::VectorXf such that later on we
// can decide to use doulbe if we want. Double vs float might give nontrivial differences..

namespace teaser {

/**
 * Struct to hold solution to a registration problem
 */
struct RegistrationSolution {
  bool valid = true;
  double scale;
  Eigen::Vector3d translation;
  Eigen::Matrix3d rotation;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * Abstract virtual class for decoupling specific scale estimation methods with interfaces.
 */
class AbstractScaleSolver {
public:
  virtual ~AbstractScaleSolver() {}

  /**
   * Pure virtual method for solving scale. Different implementations may have different assumptions
   * about input data.
   * @param src
   * @param dst
   * @return estimated scale (s)
   */
  virtual void solveForScale(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                             const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst, double* scale,
                             Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers) = 0;
};

/**
 * Abstract virtual class for decoupling specific rotation estimation method implementations with
 * interfaces.
 */
class AbstractRotationSolver {
public:
  virtual ~AbstractRotationSolver() {}

  /**
   * Pure virtual method for solving rotation. Different implementations may have different
   * assumptions about input data.
   * @param src
   * @param dst
   * @return estimated rotation matrix (R)
   */
  virtual void solveForRotation(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                                const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                                Eigen::Matrix3d* rotation,
                                Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers) = 0;
};

/**
 * Abstract virtual class for decoupling specific translation estimation method implementations with
 * interfaces.
 */
class AbstractTranslationSolver {
public:
  virtual ~AbstractTranslationSolver() {}

  /**
   * Pure virtual method for solving translation. Different implementations may have different
   * assumptions about input data.
   * @param src
   * @param dst
   * @return estimated translation vector
   */
  virtual void solveForTranslation(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                                   const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                                   Eigen::Vector3d* translation,
                                   Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers) = 0;
};

/**
 * Performs scalar truncated least squares estimation
 */
class ScalarTLSEstimator {
public:
  ScalarTLSEstimator() = default;
  /**
   * Use truncated least squares method to estimate true x given measurements X
   * TODO: call measurements Z or Y to avoid confusion with x
   * TODO: specify which type/size is x and X in the comments
   * @param X Available measurements
   * @param ranges Maximum admissible errors for measurements X
   * @param estimate (output) pointer to a double holding the estimate
   * @param inliers (output) pointer to a Eigen row vector of inliers
   */
  void estimate(const Eigen::RowVectorXd& X, const Eigen::RowVectorXd& ranges, double* estimate,
                Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers);

  /**
   * A slightly different implementation of TLS estimate. Use loop tiling to achieve potentially
   * faster performance.
   * @param X Available measurements
   * @param ranges Maximum admissible errors for measurements X
   * @param s scale for tiling
   * @param estimate (output) pointer to a double holding the estimate
   * @param inliers (output) pointer to a Eigen row vector of inliers
   */
  void estimate_tiled(const Eigen::RowVectorXd& X, const Eigen::RowVectorXd& ranges, const int& s,
                      double* estimate, Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers);
};

/**
 * Perform scale estimation using truncated least-squares (TLS)
 */
class TLSScaleSolver : public AbstractScaleSolver {
public:
  TLSScaleSolver() = delete;

  explicit TLSScaleSolver(double noise_bound, double cbar2)
      : noise_bound_(noise_bound), cbar2_(cbar2) {
    assert(noise_bound > 0);
    assert(cbar2 > 0);
  };

  /**
   * Use TLS (adaptive voting) to solve for scale. Assume dst = s * R * src
   * @param src
   * @param dst
   * @return a double indicating the estimated scale
   */
  void solveForScale(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                     const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst, double* scale,
                     Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers) override;

private:
  double noise_bound_;
  double cbar2_; // maximal allowed residual^2 to noise bound^2 ratio
  ScalarTLSEstimator tls_estimator_;
};

/**
 * Perform outlier pruning / inlier selection based on scale. This class does not perform scale
 * estimation. Rather, it estimates outliers based on the assumption that there is no scale
 * difference between the two provided vector of points.
 */
class ScaleInliersSelector : public AbstractScaleSolver {
public:
  ScaleInliersSelector() = delete;

  explicit ScaleInliersSelector(double noise_bound, double cbar2)
      : noise_bound_(noise_bound), cbar2_(cbar2){};
  /**
   * Assume dst = src + noise. The scale output will always be set to 1.
   * @param src [in] a vector of points
   * @param dst [in] a vector of points
   * @param scale [out] a constant of 1
   * @param inliers [out] a row vector of booleans indicating whether a measurement is an inlier
   */
  void solveForScale(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                     const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst, double* scale,
                     Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers) override;

private:
  double noise_bound_;
  double cbar2_; // maximal allowed residual^2 to noise bound^2 ratio
};

/**
 * Perform translation estimation using truncated least-squares (TLS)
 */
class TLSTranslationSolver : public AbstractTranslationSolver {
public:
  TLSTranslationSolver() = delete;

  explicit TLSTranslationSolver(double noise_bound, double cbar2)
      : noise_bound_(noise_bound), cbar2_(cbar2){};

  /**
   * Estimate translation between src and dst points. Assume dst = src + t.
   * @param src
   * @param dst
   * @param translation output parameter for the translation vector
   * @param inliers output parameter for detected outliers
   */
  void solveForTranslation(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                           const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                           Eigen::Vector3d* translation,
                           Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers) override;

private:
  double noise_bound_;
  double cbar2_; // maximal allowed residual^2 to noise bound^2 ratio
  ScalarTLSEstimator tls_estimator_;
};

/**
 * Base class for GNC-based rotation solvers
 */
class GNCRotationSolver : public AbstractRotationSolver {

public:
  struct Params {
    size_t max_iterations;
    double cost_threshold;
    double gnc_factor;
    double noise_bound;
  };

  GNCRotationSolver(Params params) : params_(params) {}

  Params getParams() { return params_; }

  void setParams(Params params) { params_ = params; }

  /**
   * Return the cost of the GNC solver at termination. Details of the cost function is dependent on
   * the specific solver implementation.
   *
   * @return cost at termination of the GNC solver. Undefined if run before running the solver.
   */
  double getCostAtTermination() { return cost_; }

protected:
  Params params_;
  double cost_;
};

/**
 * Use GNC-TLS to solve rotation estimation problems.
 *
 * For more information, please refer to:
 * H. Yang, P. Antonante, V. Tzoumas, and L. Carlone, “Graduated Non-Convexity for Robust Spatial
 * Perception: From Non-Minimal Solvers to Global Outlier Rejection,” arXiv:1909.08605 [cs, math],
 * Sep. 2019.
 */
class GNCTLSRotationSolver : public GNCRotationSolver {
public:
  GNCTLSRotationSolver() = delete;

  /**
   * Parametrized constructor
   * @param params
   */
  explicit GNCTLSRotationSolver(Params params) : GNCRotationSolver(params){};

  /**
   * Estimate rotation between src & dst using GNC-TLS method
   * @param src
   * @param dst
   * @param rotation
   * @param inliers
   */
  void solveForRotation(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                        const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                        Eigen::Matrix3d* rotation,
                        Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers) override;
};

/**
 * Use Fast Global Registration to solve for pairwise registration problems
 *
 * For more information, please see the original paper on FGR:
 * Q.-Y. Zhou, J. Park, and V. Koltun, “Fast Global Registration,” in Computer Vision – ECCV 2016,
 * Cham, 2016, vol. 9906, pp. 766–782.
 * Notice that our implementation differ from the paper on the estimation of T matrix. We
 * only estimate rotation, instead of rotation and translation.
 *
 */
class FastGlobalRegistrationSolver : public GNCRotationSolver {
public:
  /**
   * Remove default constructor
   */
  FastGlobalRegistrationSolver() = delete;

  /**
   * Parametrized constructor
   * @param params
   * @param rotation_only
   */
  explicit FastGlobalRegistrationSolver(Params params) : GNCRotationSolver(params){};

  /**
   * Solve a pairwise registration problem given two sets of points.
   * Notice that we assume no scale difference between v1 & v2.
   * @param src
   * @param dst
   * @return a RegistrationSolution struct.
   */
  void solveForRotation(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                        const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                        Eigen::Matrix3d* rotation,
                        Eigen::Matrix<bool, 1, Eigen::Dynamic>* inliers) override;
};

/**
 * Solve registration problems robustly.
 *
 * For more information, please refer to:
 * H. Yang, J. Shi, and L. Carlone, “TEASER: Fast and Certifiable Point Cloud Registration,”
 * arXiv:2001.07715 [cs, math], Jan. 2020.
 */
class RobustRegistrationSolver {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * An enum class representing the available GNC rotation estimation algorithms.
   *
   * GNC_TLS: see H. Yang, P. Antonante, V. Tzoumas, and L. Carlone, “Graduated Non-Convexity for
   * Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection,”
   * arXiv:1909.08605 [cs, math], Sep. 2019.
   *
   * FGR: see Q.-Y. Zhou, J. Park, and V. Koltun, “Fast Global Registration,” in Computer Vision –
   * ECCV 2016, Cham, 2016, vol. 9906, pp. 766–782. and H. Yang, P. Antonante, V. Tzoumas, and L.
   * Carlone, “Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to
   * Global Outlier Rejection,” arXiv:1909.08605 [cs, math], Sep. 2019.
   */
  enum class ROTATION_ESTIMATION_ALGORITHM {
    GNC_TLS = 0,
    FGR = 1,
  };

  /**
   * Enum representing the type of graph-based inlier selection algorithm to use
   *
   * PMC_EXACT: Use PMC to find exact clique from the inlier graph
   * PMC_HEU: Use PMC's heuristic finder to find approximate max clique
   * KCORE_HEU: Use k-core heuristic to select inliers
   * NONE: No inlier selection
   */
  enum class INLIER_SELECTION_MODE {
    PMC_EXACT = 0,
    PMC_HEU = 1,
    KCORE_HEU = 2,
    NONE = 3,
  };

  /**
   * Enum representing the formulation of the TIM graph after maximum clique filtering.
   *
   * CHAIN: formulate TIMs by only calculating the TIMs for consecutive measurements
   * COMPLETE: formulate a fully connected TIM graph
   */
  enum class INLIER_GRAPH_FORMULATION {
    CHAIN = 0,
    COMPLETE = 1,
  };

  /**
   * A struct representing params for initializing the RobustRegistrationSolver
   *
   * Note: the default values needed to be changed accordingly for best performance.
   */
  struct Params {

    /**
     * A bound on the noise of each provided measurement.
     */
    double noise_bound = 0.01;

    /**
     * Square of the ratio between acceptable noise and noise bound. Usually set to 1.
     */
    double cbar2 = 1;

    /**
     * Whether the scale is known. If set to False, the solver assumes no scale differences
     * between the src and dst points. If set to True, the solver will first solve for scale.
     *
     * When the solver does not estimate scale, it solves the registration problem much faster.
     */
    bool estimate_scaling = true;

    /**
     * Which algorithm to use to estimate rotations.
     */
    ROTATION_ESTIMATION_ALGORITHM rotation_estimation_algorithm =
        ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;

    /**
     * Factor to multiple/divide the control parameter in the GNC algorithm.
     *
     * For FGR: the algorithm divides the control parameter by the factor every iteration.
     * For GNC-TLS: the algorithm multiples the control parameter by the factor every iteration.
     */
    double rotation_gnc_factor = 1.4;

    /**
     * Maximum iterations allowed for the GNC rotation estimators.
     */
    size_t rotation_max_iterations = 100;

    /**
     * Cost threshold for the GNC rotation estimators.
     *
     * For FGR / GNC-TLS algorithm, the cost thresholds represent different meanings.
     * For FGR: the cost threshold compares with the computed cost at each iteration
     * For GNC-TLS: the cost threshold compares with the difference between costs of consecutive
     * iterations.
     */
    double rotation_cost_threshold = 1e-6;

    /**
     * Type of TIM graph given to GNC rotation solver
     */
    INLIER_GRAPH_FORMULATION rotation_tim_graph = INLIER_GRAPH_FORMULATION::CHAIN;

    /**
     * \brief Type of the inlier selection
     */
    INLIER_SELECTION_MODE inlier_selection_mode = INLIER_SELECTION_MODE::PMC_EXACT;

    /**
     * \brief The threshold ratio for determining whether to skip max clique and go straightly to
     * GNC rotation estimation. Set this to 1 to always use exact max clique selection, 0 to always
     * skip exact max clique selection.
     *
     * \attention Note that the use_max_clique option takes precedence. In other words, if
     * use_max_clique is set to false, then kcore_heuristic_threshold will be ignored. If
     * use_max_clique is set to true, then the following will happen: if the max core number of the
     * inlier graph is lower than the kcore_heuristic_threshold as a percentage of the total nodes
     * in the inlier graph, then the code will preceed to call the max clique finder. Otherwise, the
     * graph will be directly fed to the GNC rotation solver.
     *
     */
    double kcore_heuristic_threshold = 0.5;

    /**
     * \deprecated Use inlier_selection_mode instead
     * Set this to true to enable max clique inlier selection, false to skip it.
     */
    bool use_max_clique = true;

    /**
     * \deprecated Use inlier_selection_mode instead
     * Set this to false to enable heuristic only max clique finding.
     */
    bool max_clique_exact_solution = true;

    /**
     * Time limit on running the max clique algorithm (in seconds).
     */
    double max_clique_time_limit = 3600;
  };

  RobustRegistrationSolver() = default;

  /**
   * A constructor that takes in parameters and initialize the estimators accordingly.
   *
   * This is the preferred way of initializing the different estimators, instead of setting
   * each estimator one by one.
   * @param params
   */
  RobustRegistrationSolver(const Params& params);

  /**
   * Given a 3-by-N matrix representing points, return Translation Invariant Measurements (TIMs)
   * @param v a 3-by-N matrix
   * @return a 3-by-(N-1)*N matrix representing TIMs
   */
  Eigen::Matrix<double, 3, Eigen::Dynamic>
  computeTIMs(const Eigen::Matrix<double, 3, Eigen::Dynamic>& v,
              Eigen::Matrix<int, 2, Eigen::Dynamic>* map);

  /**
   * Solve for scale, translation and rotation.
   *
   * @param src_cloud source point cloud (to be transformed)
   * @param dst_cloud target point cloud (after transformation)
   * @param correspondences A vector of tuples representing the correspondences between pairs of
   * points in the two clouds
   */
  RegistrationSolution solve(const teaser::PointCloud& src_cloud,
                             const teaser::PointCloud& dst_cloud,
                             const std::vector<std::pair<int, int>> correspondences);

  /**
   * Solve for scale, translation and rotation. Assumes v2 is v1 after transformation.
   * @param v1
   * @param v2
   */
  RegistrationSolution solve(const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                             const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst);

  /**
   * Solve for scale. Assume v2 = s * R * v1, this function estimates s.
   * @param v1
   * @param v2
   */
  double solveForScale(const Eigen::Matrix<double, 3, Eigen::Dynamic>& v1,
                       const Eigen::Matrix<double, 3, Eigen::Dynamic>& v2);

  /**
   * Solve for translation.
   * @param v1
   * @param v2
   */
  Eigen::Vector3d solveForTranslation(const Eigen::Matrix<double, 3, Eigen::Dynamic>& v1,
                                      const Eigen::Matrix<double, 3, Eigen::Dynamic>& v2);

  /**
   * Solve for rotation. Assume v2 = R * v1, this function estimates find R.
   * @param v1
   * @param v2
   */
  Eigen::Matrix3d solveForRotation(const Eigen::Matrix<double, 3, Eigen::Dynamic>& v1,
                                   const Eigen::Matrix<double, 3, Eigen::Dynamic>& v2);

  /**
   * Return the cost at termination of the GNC rotation solver. Can be used to
   * assess the quality of the solution.
   *
   * @return cost at termination of the GNC solver. Undefined if run before running the solver.
   */
  inline double getGNCRotationCostAtTermination() {
    return rotation_solver_->getCostAtTermination();
  }

  /**
   * Return the solution to the registration problem.
   * @return
   */
  inline RegistrationSolution getSolution() { return solution_; };

  /**
   * Set the scale estimator used
   * @param estimator
   */
  inline void setScaleEstimator(std::unique_ptr<AbstractScaleSolver> estimator) {
    scale_solver_ = std::move(estimator);
  }

  /**
   * Set the rotation estimator used.
   *
   * Note: due to the fact that rotation estimator takes in a noise bound that is dependent on the
   * estimated scale, make sure to set the correct params before calling solve.
   * @param estimator
   */
  inline void setRotationEstimator(std::unique_ptr<GNCRotationSolver> estimator) {
    rotation_solver_ = std::move(estimator);
  }

  /**
   * Set the translation estimator used.
   * @param estimator
   */
  inline void setTranslationEstimator(std::unique_ptr<AbstractTranslationSolver> estimator) {
    translation_solver_ = std::move(estimator);
  }

  /**
   * Return a boolean Eigen row vector indicating whether specific measurements are inliers
   * according to scales.
   *
   * @return a 1-by-(number of TIMs) boolean Eigen matrix
   */
  inline Eigen::Matrix<bool, 1, Eigen::Dynamic> getScaleInliersMask() {
    return scale_inliers_mask_;
  }

  /**
   * Return the index map for scale inliers (equivalent to the index map for TIMs).
   *
   * @return a 2-by-(number of TIMs) Eigen matrix. Entries in one column represent the indices of
   * the two measurements used to calculate the corresponding TIM.
   */
  inline Eigen::Matrix<int, 2, Eigen::Dynamic> getScaleInliersMap() { return src_tims_map_; }

  /**
   * Return inlier TIMs from scale estimation
   *
   * @return a vector of tuples. Entries in each tuple represent the indices of
   * the two measurements used to calculate the corresponding TIM: measurement at indice 0 minus
   * measurement at indice 1.
   */
  inline std::vector<std::tuple<int, int>> getScaleInliers() {
    std::vector<std::tuple<int, int>> result;
    for (size_t i = 0; i < scale_inliers_mask_.cols(); ++i) {
      if (scale_inliers_mask_(i)) {
        result.emplace_back(src_tims_map_(0, i), src_tims_map_(1, i));
      }
    }
    return result;
  }

  /**
   * Return a boolean Eigen row vector indicating whether specific measurements are inliers
   * according to the rotation solver.
   *
   * @return a 1-by-(size of TIMs used in rotation estimation) boolean Eigen matrix. It is
   * equivalent to a binary mask on the TIMs used in rotation estimation, with true representing
   * that the measurement is an inlier after rotation estimation.
   */
  inline Eigen::Matrix<bool, 1, Eigen::Dynamic> getRotationInliersMask() {
    return rotation_inliers_mask_;
  }

  /**
   * Return the index map for translation inliers (equivalent to max clique).
   * /TODO: This is obsolete now. Remove or update
   *
   * @return a 1-by-(size of max clique) Eigen matrix. Entries represent the indices of the original
   * measurements.
   */
  inline Eigen::Matrix<int, 1, Eigen::Dynamic> getRotationInliersMap() {
    Eigen::Matrix<int, 1, Eigen::Dynamic> map = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic>>(
        max_clique_.data(), 1, max_clique_.size());
    return map;
  }

  /**
   * Return inliers from rotation estimation
   *
   * @return a vector of indices of TIMs passed to rotation estimator deemed as inliers by rotation
   * estimation. Note that depending on the rotation_tim_graph parameter, number of TIMs passed to
   * rotation estimation will be different.
   */
  inline std::vector<int> getRotationInliers() { return rotation_inliers_; }

  /**
   * Return a boolean Eigen row vector indicating whether specific measurements are inliers
   * according to translation measurements.
   *
   * @return a 1-by-(size of max clique) boolean Eigen matrix. It is equivalent to a binary mask on
   * the inlier max clique, with true representing that the measurement is an inlier after
   * translation estimation.
   */
  inline Eigen::Matrix<bool, 1, Eigen::Dynamic> getTranslationInliersMask() {
    return translation_inliers_mask_;
  }

  /**
   * Return the index map for translation inliers (equivalent to max clique).
   *
   * @return a 1-by-(size of max clique) Eigen matrix. Entries represent the indices of the original
   * measurements.
   */
  inline Eigen::Matrix<int, 1, Eigen::Dynamic> getTranslationInliersMap() {
    Eigen::Matrix<int, 1, Eigen::Dynamic> map = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic>>(
        max_clique_.data(), 1, max_clique_.size());
    return map;
  }

  /**
   * Return inliers from rotation estimation
   *
   * @return a vector of indices of measurements deemed as inliers by rotation estimation
   */
  inline std::vector<int> getTranslationInliers() { return translation_inliers_; }

  /**
   * Return a boolean Eigen row vector indicating whether specific measurements are inliers
   * according to translation measurements.
   * @return
   */
  inline std::vector<int> getInlierMaxClique() { return max_clique_; }

  inline std::vector<std::vector<int>> getInlierGraph() { return inlier_graph_.getAdjList(); }

  /**
   * Get TIMs built from source point cloud.
   * @return
   */
  inline Eigen::Matrix<double, 3, Eigen::Dynamic> getSrcTIMs() { return src_tims_; }

  /**
   * Get TIMs built from target point cloud.
   * @return
   */
  inline Eigen::Matrix<double, 3, Eigen::Dynamic> getDstTIMs() { return dst_tims_; }

  /**
   * Get src TIMs built after max clique pruning.
   * @return
   */
  inline Eigen::Matrix<double, 3, Eigen::Dynamic> getMaxCliqueSrcTIMs() { return pruned_src_tims_; }

  /**
   * Get dst TIMs built after max clique pruning.
   * @return
   */
  inline Eigen::Matrix<double, 3, Eigen::Dynamic> getMaxCliqueDstTIMs() { return pruned_dst_tims_; }

  /**
   * Get the index map of the TIMs built from source point cloud.
   * @return
   */
  inline Eigen::Matrix<int, 2, Eigen::Dynamic> getSrcTIMsMap() { return src_tims_map_; }

  /**
   * Get the index map of the TIMs built from target point cloud.
   * @return
   */
  inline Eigen::Matrix<int, 2, Eigen::Dynamic> getDstTIMsMap() { return dst_tims_map_; }

  /**
   * Get the index map of the TIMs used in rotation estimation.
   * @return
   */
  inline Eigen::Matrix<int, 2, Eigen::Dynamic> getSrcTIMsMapForRotation() {
    return src_tims_map_rotation_;
  }

  /**
   * Get the index map of the TIMs used in rotation estimation.
   * @return
   */
  inline Eigen::Matrix<int, 2, Eigen::Dynamic> getDstTIMsMapForRotation() {
    return dst_tims_map_rotation_;
  }

  /**
   * Reset the solver using the provided params
   * @param params a Params struct
   */
  void reset(const Params& params) {
    params_ = params;

    // Initialize the scale estimator
    if (params_.estimate_scaling) {
      setScaleEstimator(
          std::make_unique<teaser::TLSScaleSolver>(params_.noise_bound, params_.cbar2));
    } else {
      setScaleEstimator(
          std::make_unique<teaser::ScaleInliersSelector>(params_.noise_bound, params_.cbar2));
    }

    // Initialize the rotation estimator
    teaser::GNCRotationSolver::Params rotation_params{
        params_.rotation_max_iterations, params_.rotation_cost_threshold,
        params_.rotation_gnc_factor, params_.noise_bound};
    switch (params_.rotation_estimation_algorithm) {
    case ROTATION_ESTIMATION_ALGORITHM::GNC_TLS: { // GNC-TLS method
      setRotationEstimator(std::make_unique<teaser::GNCTLSRotationSolver>(rotation_params));
      break;
    }
    case ROTATION_ESTIMATION_ALGORITHM::FGR: { // FGR method
      setRotationEstimator(std::make_unique<teaser::FastGlobalRegistrationSolver>(rotation_params));
      break;
    }
    }

    // Initialize the translation estimator
    setTranslationEstimator(
        std::make_unique<teaser::TLSTranslationSolver>(params_.noise_bound, params_.cbar2));

    // Clear member variables
    max_clique_.clear();
    rotation_inliers_.clear();
    translation_inliers_.clear();
    inlier_graph_.clear();
  }

  /**
   * Return the params
   * @return a Params struct
   */
  Params getParams() { return params_; }

private:
  Params params_;
  RegistrationSolution solution_;

  // Inlier Binary Vectors
  Eigen::Matrix<bool, 1, Eigen::Dynamic> scale_inliers_mask_;
  Eigen::Matrix<bool, 1, Eigen::Dynamic> rotation_inliers_mask_;
  Eigen::Matrix<bool, 1, Eigen::Dynamic> translation_inliers_mask_;

  // TIMs
  // TIMs used for scale estimation/pruning
  Eigen::Matrix<double, 3, Eigen::Dynamic> src_tims_;
  Eigen::Matrix<double, 3, Eigen::Dynamic> dst_tims_;
  // TIMs used for rotation estimation
  Eigen::Matrix<double, 3, Eigen::Dynamic> pruned_src_tims_;
  Eigen::Matrix<double, 3, Eigen::Dynamic> pruned_dst_tims_;

  // TIM maps
  // for scale estimation
  Eigen::Matrix<int, 2, Eigen::Dynamic> src_tims_map_;
  Eigen::Matrix<int, 2, Eigen::Dynamic> dst_tims_map_;
  // for rotation estimation
  Eigen::Matrix<int, 2, Eigen::Dynamic> src_tims_map_rotation_;
  Eigen::Matrix<int, 2, Eigen::Dynamic> dst_tims_map_rotation_;

  // Max clique vector
  std::vector<int> max_clique_;

  // Inliers after rotation estimation
  std::vector<int> rotation_inliers_;

  // Inliers after translation estimation (final inliers)
  std::vector<int> translation_inliers_;

  // Inlier graph
  teaser::Graph inlier_graph_;

  // Ptrs to Solvers
  std::unique_ptr<AbstractScaleSolver> scale_solver_;
  std::unique_ptr<GNCRotationSolver> rotation_solver_;
  std::unique_ptr<AbstractTranslationSolver> translation_solver_;
};

} // namespace teaser
