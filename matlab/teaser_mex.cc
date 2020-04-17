/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <map>
#include <iostream>
#include <chrono>

#include "mex.h"
#include <Eigen/Core>

#include "teaser_mex_utils.h"
#include "teaser/registration.h"

enum class INPUT_PARAMS : int {
  src = 0,
  dst = 1,
  cbar2 = 2,
  noise_bound = 3,
  estimate_scaling = 4,
  rotation_estimation_algorithm = 5,
  rotation_gnc_factor = 6,
  rotation_max_iterations = 7,
  rotation_cost_threshold = 8,
  inlier_selection_algorithm = 9,
  kcore_heuristic_threshold = 10,
};

enum class OUTPUT_PARAMS : int {
  s_est = 0,
  R_est = 1,
  t_est = 2,
  time_taken = 3,
};

typedef bool (*mexTypeCheckFunction)(const mxArray*);
const std::map<INPUT_PARAMS, mexTypeCheckFunction> INPUT_PARMS_MAP{
    {INPUT_PARAMS::src, &isPointCloudMatrix},
    {INPUT_PARAMS::dst, &isPointCloudMatrix},
    {INPUT_PARAMS::cbar2, &isRealDoubleScalar},
    {INPUT_PARAMS::noise_bound, &isRealDoubleScalar},
    {INPUT_PARAMS::estimate_scaling, &mxIsLogicalScalar},
    {INPUT_PARAMS::rotation_estimation_algorithm, &isRealDoubleScalar},
    {INPUT_PARAMS::rotation_gnc_factor, &isRealDoubleScalar},
    {INPUT_PARAMS::rotation_max_iterations, &isRealDoubleScalar},
    {INPUT_PARAMS::rotation_cost_threshold, &isRealDoubleScalar},
    {INPUT_PARAMS::inlier_selection_algorithm, &isRealDoubleScalar},
    {INPUT_PARAMS::kcore_heuristic_threshold, &isRealDoubleScalar},
};
const std::map<OUTPUT_PARAMS, mexTypeCheckFunction> OUTPUT_PARMS_MAP{
    {OUTPUT_PARAMS::s_est, &isRealDoubleScalar},
    {OUTPUT_PARAMS::R_est, &isRealDoubleMatrix<3, 3>},
    {OUTPUT_PARAMS::t_est, &isRealDoubleMatrix<3, 1>},
    {OUTPUT_PARAMS::time_taken, &isRealDoubleScalar},
};

/**
 * This is the MATLAB binding for TEASER++.
 *
 * Input:
 * - src: a 3-by-N matrix of 3D points representing points to be transformed
 * - dst: a 3-by-N matrix of 3D points representing points after transformation
 * - cbar2: square of maximum allowed ratio between noise and noise bound (see [1]).
 * - noise_bound: a floating-point number indicating the bound on noise
 * - estimate_scaling: a boolean indicating whether scale needs to be estimated
 * - rotation_max_iterations: maximum iterations for the rotation estimation loop
 * - rotation_cost_threshold: cost threshold for rotation termination
 * - rotation_gnc_factor: gnc factor for rotation estimation
 *                        for GNC-TLS method: it's multiplied on the GNC control parameter
 *                        for FGR method: it's divided on the GNC control parameter
 * - rotation_estimation_algorithm: a number indicating the rotation estimation method used;
 *                                  if it's 0: GNC-TLS
 *                                  if it's 1: FGR
 * - inlier_selection_algorithm: a number indicating the  method used;
 *                                  0: PMC_EXACT
 *                                  1: PMC_HEU
 *                                  2: KCORE_HEU
 *                                  3: NONE
 * - kcore_heuristic_threshold: threshold for k-core heuristic. If the inlier graph has a max core
 *                              number greater than kcore_heuristic_threshold * number of vertices,
 *                              then the nodes with core number == max core number are directly
 *                              returned. If not, then the algorithm will proceed to use PMC to find
 *                              an exact clique.
 *
 * Output:
 * - s_est estimated scale (scalar)
 * - R_est estimated rotation matrix (3-by-3 matrix)
 * - t_est estimated translation vector (3-by-1 matrix)
 * - time_taken time it takes for the underlying TEASER++ library to compute a solution.
 *
 * [1] H. Yang, J. Shi, and L. Carlone, “TEASER: Fast and Certifiable Point Cloud Registration,”
 * arXiv:2001.07715 [cs, math], Jan. 2020.
 *
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

  // Check for proper number of arguments
  if (nrhs != INPUT_PARMS_MAP.size()) {
    mexErrMsgIdAndTxt("teaserSolve:nargin", "Wrong number of input arguments.");
  }
  if (nlhs != OUTPUT_PARMS_MAP.size()) {
    mexErrMsgIdAndTxt("teaserSolve:nargin", "Wrong number of output arguments.");
  }

  // Check for proper input types
  for (const auto& pair : INPUT_PARMS_MAP) {
    if (!pair.second(prhs[toUType(pair.first)])) {
      std::stringstream error_msg;
      error_msg << "Argument " << toUType(pair.first) + 1 << " has the wrong type.\n";
      mexErrMsgIdAndTxt("teaserSolve:nargin", error_msg.str().c_str());
    }
  }

  mexPrintf("Arguments type checks passed.\n");
  mexEvalString("drawnow;");

  // Prepare parameters
  // Prepare source and destination Eigen point matrices
  Eigen::Matrix<double, 3, Eigen::Dynamic> src_eigen, dst_eigen;
  mexPointMatrixToEigenMatrix(prhs[toUType(INPUT_PARAMS::src)], &src_eigen);
  mexPointMatrixToEigenMatrix(prhs[toUType(INPUT_PARAMS::dst)], &dst_eigen);

  // Other parameters
  auto cbar2 = static_cast<double>(*mxGetPr(prhs[toUType(INPUT_PARAMS::cbar2)]));
  auto noise_bound = static_cast<double>(*mxGetPr(prhs[toUType(INPUT_PARAMS::noise_bound)]));
  auto estimate_scaling =
      static_cast<bool>(*mxGetPr(prhs[toUType(INPUT_PARAMS::estimate_scaling)]));
  auto rotation_estimation_method =
      static_cast<int>(*mxGetPr(prhs[toUType(INPUT_PARAMS::rotation_estimation_algorithm)]));
  auto rotation_gnc_factor =
      static_cast<double>(*mxGetPr(prhs[toUType(INPUT_PARAMS::rotation_gnc_factor)]));
  auto rotation_max_iterations =
      static_cast<size_t>(*mxGetPr(prhs[toUType(INPUT_PARAMS::rotation_max_iterations)]));
  auto rotation_cost_threshold =
      static_cast<double>(*mxGetPr(prhs[toUType(INPUT_PARAMS::rotation_cost_threshold)]));
  auto inlier_selection_algorithm =
      static_cast<int>(*mxGetPr(prhs[toUType(INPUT_PARAMS::inlier_selection_algorithm)]));
  auto kcore_heuristic_threshold =
      static_cast<double>(*mxGetPr(prhs[toUType(INPUT_PARAMS::kcore_heuristic_threshold)]));

  // Prepare the TEASER++ solver for solving registration problem
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = noise_bound;
  params.cbar2 = cbar2;
  params.estimate_scaling = estimate_scaling;
  params.rotation_max_iterations = rotation_max_iterations;
  params.rotation_gnc_factor = rotation_gnc_factor;
  params.rotation_cost_threshold = rotation_cost_threshold;
  params.kcore_heuristic_threshold = kcore_heuristic_threshold;

  switch (rotation_estimation_method) {
  case 0: { // GNC-TLS method
    mexPrintf("Use GNC-TLS for rotation estimation.\n");
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    break;
  }
  case 1: { // FGR method
    mexPrintf("Use FGR for rotation estimation.\n");
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
    break;
  }
  default: {
    mexPrintf("Rotation estimation method given does not exist. Use GNC-TLS instead.\n");
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    break;
  }
  }

  switch (inlier_selection_algorithm) {
  case 0: { // PMC_EXACT method
    mexPrintf("Use PMC_EXACT for inlier selection.\n");
    params.inlier_selection_mode =
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
    break;
  }
  case 1: { // PMC_HEU method
    mexPrintf("Use PMC_HEU for inlier selection.\n");
    params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;
    break;
  }
  case 2: { // KCORE_HEU method
    mexPrintf("Use KCORE_HEU for inlier selection.\n");
    params.inlier_selection_mode =
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::KCORE_HEU;
    break;
  }
  case 3: { // NONE
    mexPrintf("No inlier selection step after scale pruning.\n");
    params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE;
    break;
  }
  default: {
    mexPrintf("Unknown inlier selection algorithm given. Use PMC_EXACT instead.\n");
    params.inlier_selection_mode =
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
    break;
  }
  }

  teaser::RobustRegistrationSolver solver(params);

  mexPrintf("Start TEASER++ solver.\n");
  mexEvalString("drawnow;");

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  // Solve
  assert(src_eigen.size() != 0);
  assert(dst_eigen.size() != 0);
  solver.solve(src_eigen, dst_eigen);

  // Stop the timer
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  double duration_in_milliseconds = static_cast<double>(duration.count()) / 1000.0;

  auto solution = solver.getSolution();

  mexPrintf("TEASER++ has found a solution in %f milliseconds.\n", duration_in_milliseconds);
  mexEvalString("drawnow;");

  // Populate outputs
  plhs[toUType(OUTPUT_PARAMS::s_est)] = mxCreateDoubleScalar(solution.scale);
  // Populate output R matrix
  plhs[toUType(OUTPUT_PARAMS::R_est)] = mxCreateDoubleMatrix(3, 3, mxREAL);
  Eigen::Map<Eigen::Matrix3d> R_map(mxGetPr(plhs[toUType(OUTPUT_PARAMS::R_est)]), 3, 3);
  R_map = solution.rotation;

  // Populate output T vector
  plhs[toUType(OUTPUT_PARAMS::t_est)] = mxCreateDoubleMatrix(3, 1, mxREAL);
  Eigen::Map<Eigen::Matrix<double, 3, 1>> t_map(mxGetPr(plhs[toUType(OUTPUT_PARAMS::t_est)]), 3, 1);
  t_map = solution.translation;

  // Populate time output
  plhs[toUType(OUTPUT_PARAMS::time_taken)] = mxCreateDoubleScalar(duration_in_milliseconds);
}
