/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <limits>
#include <fstream>
#include <chrono>
#include <algorithm>

#include <teaser/macros.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>

#include "teaser/certification.h"
#include "teaser/linalg.h"

teaser::CertificationResult
teaser::DRSCertifier::certify(const Eigen::Matrix3d& R_solution,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                              const Eigen::Matrix<bool, 1, Eigen::Dynamic>& theta) {
  // convert theta to a double Eigen matrix
  Eigen::Matrix<double, 1, Eigen::Dynamic> theta_double(1, theta.cols());
  for (size_t i = 0; i < theta.cols(); ++i) {
    if (theta(i)) {
      theta_double(i) = 1;
    } else {
      theta_double(i) = -1;
    }
  }
  return certify(R_solution, src, dst, theta_double);
}

teaser::CertificationResult
teaser::DRSCertifier::certify(const Eigen::Matrix3d& R_solution,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                              const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta) {
  int N = src.cols();
  int Npm = 4 + 4 * N;

  // prepend theta with 1
  Eigen::Matrix<double, 1, Eigen::Dynamic> theta_prepended(1, theta.cols() + 1);
  theta_prepended << 1, theta;

  // get the inverse map
  TEASER_INFO_MSG("Starting linear inverse map calculation.\n");
  TEASER_DEBUG_DECLARE_TIMING(LProj);
  TEASER_DEBUG_START_TIMING(LProj);
  SparseMatrix inverse_map;
  getLinearProjection(theta_prepended, &inverse_map);
  TEASER_DEBUG_STOP_TIMING(LProj);
  TEASER_DEBUG_INFO_MSG("Obtained linear inverse map.");
  TEASER_DEBUG_INFO_MSG("Linear projection time: " << TEASER_DEBUG_GET_TIMING(LProj));

  // recall data matrix from QUASAR
  Eigen::MatrixXd Q_cost(Npm, Npm);
  getQCost(src, dst, &Q_cost);
  TEASER_DEBUG_INFO_MSG("Obtained Q_cost matrix.");

  // convert the estimated rotation to quaternion
  Eigen::Quaterniond q_solution(R_solution);
  q_solution.normalize();
  Eigen::VectorXd q_solution_vec(4, 1);
  q_solution_vec << q_solution.x(), q_solution.y(), q_solution.z(), q_solution.w();

  // this would have been the rank-1 decomposition of Z if Z were the globally
  // optimal solution of the QUASAR SDP
  Eigen::VectorXd x =
      teaser::vectorKron<double, Eigen::Dynamic, Eigen::Dynamic>(theta_prepended, q_solution_vec);

  // build the "rotation matrix" D_omega
  Eigen::MatrixXd D_omega;
  getBlockDiagOmega(Npm, q_solution, &D_omega);
  Eigen::MatrixXd Q_bar = D_omega.transpose() * (Q_cost * D_omega);
  Eigen::VectorXd x_bar = D_omega.transpose() * x;
  TEASER_DEBUG_INFO_MSG("Obtained D_omega matrix.");

  // build J_bar matrix with a 4-by-4 identity at the top left corner
  Eigen::SparseMatrix<double> J_bar(Npm, Npm);
  for (size_t i = 0; i < 4; ++i) {
    J_bar.insert(i, i) = 1;
  }

  // verify optimality in the "rotated" space using projection
  // this is the cost of the primal, when strong duality holds, mu is also the cost of the dual
  double mu = x.transpose().dot(Q_cost * x);

  // get initial guess
  SparseMatrix lambda_bar_init;
  getLambdaGuess(R_solution, theta, src, dst, &lambda_bar_init);

  // this initial guess lives in the affine subspace
  // use 2 separate steps to limit slow evaluation on only the few non-zeros in the sparse matrix
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
  Eigen::SparseMatrix<double> M_init = Q_bar - mu * J_bar - lambda_bar_init;
#else
  // fix for this bug in Eigen 3.2: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=632
  Eigen::SparseMatrix<double> M_init = Q_bar.sparseView() - mu * J_bar - lambda_bar_init;
#endif

  // flag to indicate whether we exceeded iterations or reach the desired sub-optim gap
  bool exceeded_maxiters = true;

  // vector to store suboptim trajectory
  std::vector<double> suboptim_traj;

  // current suboptimality gap
  double current_suboptim = std::numeric_limits<double>::infinity();
  double best_suboptim = std::numeric_limits<double>::infinity();

  // preallocate some matrices
  Eigen::MatrixXd M_PSD;
  // TODO: Make M a sparse matrix
  Eigen::MatrixXd M = M_init.toDense();
  Eigen::MatrixXd temp_W(M.rows(), M.cols());
  Eigen::MatrixXd W_dual(Npm, Npm);
  Eigen::MatrixXd M_affine(Npm, Npm);

  TEASER_INFO_MSG("Starting Douglas-Rachford Splitting.\n");
  for (size_t iter = 0; iter < params_.max_iterations; ++iter) {
    // print out iteration every 10 iteration
    TEASER_INFO_MSG_THROTTLE("Iteration: " << iter << "\n", iter, 10);

    // to nearest PSD
    TEASER_DEBUG_DECLARE_TIMING(PSD);
    TEASER_DEBUG_START_TIMING(PSD);
    teaser::getNearestPSD<double>(M, &M_PSD);
    TEASER_DEBUG_STOP_TIMING(PSD);
    TEASER_DEBUG_INFO_MSG("PSD time: " << TEASER_DEBUG_GET_TIMING(PSD));

    // projection to affine space
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
    temp_W = 2 * M_PSD - M - M_init;
#else
    // fix for this bug in Eigen 3.2: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=632
    temp_W = 2 * M_PSD - M - M_init.toDense();
#endif

    TEASER_DEBUG_DECLARE_TIMING(DualProjection);
    TEASER_DEBUG_START_TIMING(DualProjection);
    getOptimalDualProjection(temp_W, theta_prepended, inverse_map, &W_dual);
    TEASER_DEBUG_STOP_TIMING(DualProjection);
    TEASER_DEBUG_INFO_MSG("Dual Projection time: " << TEASER_DEBUG_GET_TIMING(DualProjection));
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
    M_affine = M_init + W_dual;
#else
    // fix for this bug in Eigen 3.2: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=632
    M_affine = M_init.toDense() + W_dual;
#endif

    // compute suboptimality gap
    TEASER_DEBUG_DECLARE_TIMING(Gap);
    TEASER_DEBUG_START_TIMING(Gap);
    current_suboptim = computeSubOptimalityGap(M_affine, mu, N);
    TEASER_DEBUG_STOP_TIMING(Gap);
    TEASER_DEBUG_INFO_MSG("Sub Optimality Gap time: " << TEASER_DEBUG_GET_TIMING(Gap));
    TEASER_DEBUG_INFO_MSG("Current sub-optimality gap: " << current_suboptim);

    // termination check and update trajectory
    suboptim_traj.push_back(current_suboptim);

    // update best optimality
    if (current_suboptim < best_suboptim) {
      best_suboptim = current_suboptim;
    }

    if (current_suboptim < params_.sub_optimality) {
      TEASER_DEBUG_INFO_MSG("Suboptimality condition reached in " << iter + 1
                                                                  << " iterations. Stopping DRS.");
      exceeded_maxiters = false;
      break;
    }

    // update M
    M += params_.gamma_tau * (M_affine - M_PSD);
  }

  // prepare results
  CertificationResult cert_result;
  cert_result.is_optimal = best_suboptim < params_.sub_optimality;
  cert_result.best_suboptimality = best_suboptim;
  cert_result.suboptimality_traj = suboptim_traj;
  return cert_result;
}

double teaser::DRSCertifier::computeSubOptimalityGap(const Eigen::MatrixXd& M, double mu, int N) {
  Eigen::MatrixXd new_M = (M + M.transpose()) / 2;

  bool successful = false;
  Eigen::VectorXd eig_vals;
  double min_eig;
  if (params_.eig_decomposition_solver == EIG_SOLVER_TYPE::EIGEN) {
    // Eigen
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(new_M);
    if (eigensolver.info() == Eigen::Success) {
      eig_vals = eigensolver.eigenvalues();
      min_eig = eig_vals.minCoeff();
      successful = true;
    }
  } else {
    // Spectra
    Spectra::DenseSymMatProd<double> op(new_M);
    Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::DenseSymMatProd<double>> eigs(
        &op, 1, 30);
    eigs.init();
    int nconv = eigs.compute();
    if (eigs.info() == Spectra::SUCCESSFUL) {
      eig_vals = eigs.eigenvalues();
      min_eig = eig_vals(0);
      successful = true;
    }
  }

  if (!successful) {
    TEASER_DEBUG_ERROR_MSG(
        "Failed to find the minimal eigenvalue for suboptimality gap calculaiton.");
    return std::numeric_limits<double>::infinity();
  }

  if (min_eig > 0) {
    // already optimal
    return 0;
  }
  return (-min_eig * (N + 1)) / mu;
}

void teaser::DRSCertifier::getQCost(const Eigen::Matrix<double, 3, Eigen::Dynamic>& v1,
                                    const Eigen::Matrix<double, 3, Eigen::Dynamic>& v2,
                                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Q) {
  int N = v1.cols();
  int Npm = 4 + 4 * N;
  double noise_bound_scaled = params_.cbar2 * std::pow(params_.noise_bound, 2);

  // coefficient matrix that maps vec(qq\tran) to vec(R)
  Eigen::Matrix<double, 9, 16> P(9, 16);
  // clang-format off
  P << 1,  0, 0, 0,  0, -1, 0, 0,  0, 0, -1, 0, 0,  0,  0, 1,
       0,  1, 0, 0,  1,  0, 0, 0,  0, 0, 0,  1, 0,  0,  1, 0,
       0,  0, 1, 0,  0,  0, 0, -1, 1, 0, 0,  0, 0,  -1, 0, 0,
       0,  1, 0, 0,  1,  0, 0, 0,  0, 0, 0, -1, 0,  0, -1, 0,
       -1, 0, 0, 0,  0,  1, 0, 0,  0, 0, -1, 0, 0,  0,  0, 1,
       0,  0, 0, 1,  0,  0, 1, 0,  0, 1, 0,  0, 1,  0,  0, 0,
       0,  0, 1, 0,  0,  0, 0, 1,  1, 0, 0,  0, 0,  1,  0, 0,
       0,  0, 0, -1, 0,  0, 1, 0,  0, 1, 0,  0, -1, 0,  0, 0,
       -1, 0, 0, 0,  0, -1, 0, 0,  0, 0, 1,  0, 0,  0,  0, 1;
  // clang-format on

  // Some temporary vectors to save intermediate matrices
  Eigen::Matrix3d temp_A;
  Eigen::Matrix<double, 16, 1> temp_B;
  Eigen::Matrix<double, 9, 1> temp_map2vec;
  Eigen::Matrix4d P_k;

  // Q1 matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Q1(Npm, Npm);
  Q1.setZero();
  for (size_t k = 0; k < N; ++k) {
    int start_idx = k * 4 + 4;

    //  P_k = reshape(P'*reshape(v2(:,k)*v1(:,k)',[9,1]),[4,4]);
    temp_A = v2.col(k) * (v1.col(k).transpose());
    temp_map2vec = Eigen::Map<Eigen::Matrix<double, 9, 1>>(temp_A.data());
    temp_B = P.transpose() * temp_map2vec;
    P_k = Eigen::Map<Eigen::Matrix4d>(temp_B.data());

    //  ck = 0.5 * ( v1(:,k)'*v1(:,k)+v2(:,k)'*v2(:,k) - barc2 );
    double ck = 0.5 * (v1.col(k).squaredNorm() + v2.col(k).squaredNorm() - noise_bound_scaled);
    Q1.block<4, 4>(0, start_idx) =
        Q1.block<4, 4>(0, start_idx) - 0.5 * P_k + ck / 2 * Eigen::Matrix4d::Identity();
    Q1.block<4, 4>(start_idx, 0) =
        Q1.block<4, 4>(start_idx, 0) - 0.5 * P_k + ck / 2 * Eigen::Matrix4d::Identity();
  }

  // Q2 matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Q2(Npm, Npm);
  Q2.setZero();
  for (size_t k = 0; k < N; ++k) {
    int start_idx = k * 4 + 4;

    //  P_k = reshape(P'*reshape(v2(:,k)*v1(:,k)',[9,1]),[4,4]);
    temp_A = v2.col(k) * (v1.col(k).transpose());
    temp_map2vec = Eigen::Map<Eigen::Matrix<double, 9, 1>>(temp_A.data());
    temp_B = P.transpose() * temp_map2vec;
    P_k = Eigen::Map<Eigen::Matrix4d>(temp_B.data());

    //  ck = 0.5 * ( v1(:,k)'*v1(:,k)+v2(:,k)'*v2(:,k) + barc2 );
    double ck = 0.5 * (v1.col(k).squaredNorm() + v2.col(k).squaredNorm() + noise_bound_scaled);
    Q2.block<4, 4>(start_idx, start_idx) =
        Q2.block<4, 4>(start_idx, start_idx) - P_k + ck * Eigen::Matrix4d::Identity();
  }

  *Q = Q1 + Q2;
}

Eigen::Matrix4d teaser::DRSCertifier::getOmega1(const Eigen::Quaterniond& q) {
  Eigen::Matrix4d omega1;
  // clang-format off
  omega1 << q.w(), -q.z(), q.y(), q.x(),
            q.z(), q.w(), -q.x(), q.y(),
            -q.y(), q.x(), q.w(), q.z(),
            -q.x(), -q.y(), -q.z(), q.w();
  // clang-format on
  return omega1;
}

void teaser::DRSCertifier::getBlockDiagOmega(
    int Npm, const Eigen::Quaterniond& q,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* D_omega) {
  D_omega->resize(Npm, Npm);
  D_omega->setZero();
  for (size_t i = 0; i < Npm / 4; ++i) {
    int start_idx = i * 4;
    D_omega->block<4, 4>(start_idx, start_idx) = getOmega1(q);
  }
}

void teaser::DRSCertifier::getOptimalDualProjection(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& W,
    const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta_prepended, const SparseMatrix& A_inv,
    Eigen::MatrixXd* W_dual) {
  // prepare some variables
  int Npm = W.rows();
  int N = Npm / 4 - 1;
  assert(theta_prepended.cols() == N + 1);

  // first project the off-diagonal blocks
  int nr_off_diag_blks = A_inv.rows();

  // Compute b_W
  Eigen::Matrix<double, Eigen::Dynamic, 3> b_W(nr_off_diag_blks, 3);
  b_W.setZero();

  int count = 0;
  for (size_t i = 0; i < N; ++i) {
    // prepare indices
    int row_idx_start = i * 4;
    int row_idx_end = i * 4 + 3;
    for (size_t j = i + 1; j < N + 1; ++j) {
      // prepare indices
      int col_idx_start = j * 4;
      int col_idx_end = j * 4 + 3;

      // current theta value calculation
      double theta_ij = theta_prepended.col(i) * theta_prepended.col(j);

      // [-theta_ij 1]
      Eigen::Matrix<double, 1, 2> temp_A;
      temp_A << -theta_ij, 1;

      // [-1 theta_ij]
      Eigen::Matrix<double, 1, 2> temp_B;
      temp_B << -1, theta_ij;

      // W([row_idx(4) col_idx(4)],row_idx(1:3))
      Eigen::Matrix<double, 1, 3> temp_C = W.block<1, 3>(row_idx_end, row_idx_start);
      Eigen::Matrix<double, 1, 3> temp_D = W.block<1, 3>(col_idx_end, row_idx_start);
      Eigen::Matrix<double, 2, 3> temp_CD;
      temp_CD << temp_C, temp_D;

      // W([row_idx(4) col_idx(4)], col_idx(1:3))
      Eigen::Matrix<double, 1, 3> temp_E = W.block<1, 3>(row_idx_end, col_idx_start);
      Eigen::Matrix<double, 1, 3> temp_F = W.block<1, 3>(col_idx_end, col_idx_start);
      Eigen::Matrix<double, 2, 3> temp_EF;
      temp_EF << temp_E, temp_F;

      // calculate the current row for b_W with the temporary variables
      Eigen::Matrix<double, 1, 3> y_b_Wt = temp_A * temp_CD + temp_B * temp_EF;

      // update b_W
      b_W.row(count) = y_b_Wt;
      count += 1;
    }
  }
  Eigen::Matrix<double, Eigen::Dynamic, 3> b_W_dual = A_inv * b_W;

  // Compute W_dual
  W_dual->resize(Npm, Npm);
  W_dual->setZero();
  count = 0;
  // declare matrices to prevent reallocation
  Eigen::Matrix4d W_ij = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d W_dual_ij = Eigen::Matrix4d::Zero();
  Eigen::Matrix<double, 3, 1> y_dual_ij = Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 4, Eigen::Dynamic> W_i(4, W.cols());
  Eigen::Matrix<double, 4, Eigen::Dynamic> W_dual_i(4, Npm);
  W_i.setZero();
  W_dual_i.setZero();
  for (size_t i = 0; i < N; ++i) {
    int row_idx_start = i * 4;
    W_i = W.block(row_idx_start, 0, 4, W.cols());

    for (size_t j = i + 1; j < N + 1; ++j) {
      int col_idx_start = j * 4;

      // take W_ij and break into top-left 3x3 and vectors
      W_ij = W_i.block(0, col_idx_start, 4, 4);
      y_dual_ij = (b_W_dual.row(count)).transpose();

      // assemble W_dual_ij
      W_dual_ij = (W_ij - W_ij.transpose()) / 2;
      W_dual_ij.block<3, 1>(0, 3) = y_dual_ij;
      W_dual_ij.block<1, 3>(3, 0) = -y_dual_ij.transpose();

      // assign W_dual_ij to W_dual_i
      W_dual_i.block<4, 4>(0, col_idx_start) = W_dual_ij;

      count += 1;
    }
    W_dual->block(row_idx_start, 0, 4, Npm) = W_dual_i;

    // clear out temporary variables
    W_dual_i.setZero();
    W_i.setZero();
  }
  Eigen::MatrixXd temp = W_dual->transpose();
  *W_dual += temp;

  // Project the diagonal blocks
  Eigen::Matrix4d W_ii = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d W_diag_mean = Eigen::Matrix4d::Zero();
  Eigen::Matrix3d W_diag_sum_33 = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < N + 1; ++i) {
    int idx_start = i * 4;
    // Eigen::Vector4d W_dual_row_sum_last_column= W_dual->middleRows<4>(idx_start).rowwise().sum();
    Eigen::Vector4d W_dual_row_sum_last_column;
    // sum 4 rows
    getBlockRowSum(*W_dual, idx_start, theta_prepended, &W_dual_row_sum_last_column);
    W_ii = W.block<4, 4>(idx_start, idx_start);
    // modify W_ii's last column/row to satisfy complementary slackness
    W_ii.block<4, 1>(0, 3) = -theta_prepended(i) * W_dual_row_sum_last_column;
    W_ii.block<1, 4>(3, 0) = -theta_prepended(i) * W_dual_row_sum_last_column.transpose();
    (*W_dual).block<4, 4>(idx_start, idx_start) = W_ii;
    W_diag_sum_33 += W_ii.topLeftCorner<3, 3>();
  }
  W_diag_mean.topLeftCorner<3, 3>() = W_diag_sum_33 / (N + 1);

  // update diagonal blocks
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> temp_A((N + 1) * W_diag_mean.rows(),
                                                               (N + 1) * W_diag_mean.cols());
  temp_A.setZero();
  for (int i = 0; i < N + 1; i++) {
    temp_A.block(i * W_diag_mean.rows(), i * W_diag_mean.cols(), W_diag_mean.rows(),
                 W_diag_mean.cols()) = W_diag_mean;
  }
  *W_dual -= temp_A;
}

void teaser::DRSCertifier::getLambdaGuess(const Eigen::Matrix<double, 3, 3>& R,
                                          const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta,
                                          const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                                          const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                                          SparseMatrix* lambda_guess) {
  int K = theta.cols();
  int Npm = 4 * K + 4;

  double noise_bound_scaled = params_.cbar2 * std::pow(params_.noise_bound, 2);

  // prepare the lambda sparse matrix output
  lambda_guess->resize(Npm, Npm);
  lambda_guess->reserve(Npm * (Npm - 1) * 2);
  lambda_guess->setZero();

  // 4-by-4 Eigen matrix to store the top left 4-by-4 block
  Eigen::Matrix<double, 4, 4> topleft_block = Eigen::Matrix4d::Zero();

  // 4-by-4 Eigen matrix to store the current 4-by-4 block
  Eigen::Matrix<double, 4, 4> current_block = Eigen::Matrix4d::Zero();

  for (size_t i = 0; i < K; ++i) {
    // hat maps for later usage
    Eigen::Matrix<double, 3, 3> src_i_hatmap = teaser::hatmap(src.col(i));
    if (theta(0, i) > 0) {
      // residual
      Eigen::Matrix<double, 3, 1> xi = R.transpose() * (dst.col(i) - R * src.col(i));
      Eigen::Matrix<double, 3, 3> xi_hatmap = teaser::hatmap(xi);

      // compute the (4,4) entry of the current block, obtained from KKT complementary slackness
      current_block(3, 3) = -0.75 * xi.squaredNorm() - 0.25 * noise_bound_scaled;

      // compute the top-left 3-by-3 block
      current_block.topLeftCorner<3, 3>() =
          src_i_hatmap * src_i_hatmap - 0.5 * (src.col(i)).dot(xi) * Eigen::Matrix3d::Identity() +
          0.5 * xi_hatmap * src_i_hatmap + 0.5 * xi * src.col(i).transpose() -
          0.75 * xi.squaredNorm() * Eigen::Matrix3d::Identity() -
          0.25 * noise_bound_scaled * Eigen::Matrix3d::Identity();

      // compute the vector part
      current_block.topRightCorner<3, 1>() = -1.5 * xi_hatmap * src.col(i);
      current_block.bottomLeftCorner<1, 3>() = (current_block.topRightCorner<3, 1>()).transpose();
    } else {
      // residual
      Eigen::Matrix<double, 3, 1> phi = R.transpose() * (dst.col(i) - R * src.col(i));
      Eigen::Matrix<double, 3, 3> phi_hatmap = teaser::hatmap(phi);

      // compute lambda_i, (4,4) entry
      current_block(3, 3) = -0.25 * phi.squaredNorm() - 0.75 * noise_bound_scaled;

      // compute E_ii, top-left 3-by-3 block
      current_block.topLeftCorner<3, 3>() =
          src_i_hatmap * src_i_hatmap - 0.5 * (src.col(i)).dot(phi) * Eigen::Matrix3d::Identity() +
          0.5 * phi_hatmap * src_i_hatmap + 0.5 * phi * src.col(i).transpose() -
          0.25 * phi.squaredNorm() * Eigen::Matrix3d::Identity() -
          0.25 * noise_bound_scaled * Eigen::Matrix3d::Identity();

      // compute x_i
      current_block.topRightCorner<3, 1>() = -0.5 * phi_hatmap * src.col(i);
      current_block.bottomLeftCorner<1, 3>() = (current_block.topRightCorner<3, 1>()).transpose();
    }

    // put the current block to the sparse triplets
    // start idx: i * 4
    // end idx: i * 4 + 3
    // assume current block is column major
    for (size_t col = 0; col < 4; ++col) {
      for (size_t row = 0; row < 4; ++row) {
        lambda_guess->insert((i + 1) * 4 + row, (i + 1) * 4 + col) = -current_block(row, col);
      }
    }

    // update the first block
    topleft_block += current_block;
  }

  // put the first block to the sparse matrix
  for (size_t col = 0; col < 4; ++col) {
    for (size_t row = 0; row < 4; ++row) {
      lambda_guess->coeffRef(row, col) += topleft_block(row, col);
    }
  }
}

void teaser::DRSCertifier::getLinearProjection(
    const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta_prepended, SparseMatrix* A_inv) {
  // number of off-diagonal entries in the inverse map
  int N0 = theta_prepended.cols() - 1;

  double y = 1.0 / (2 * static_cast<double>(N0) + 6);
  // number of diagonal entries in the inverse map
  double x = (static_cast<double>(N0) + 1.0) * y;

  int N = N0 + 1;

  // build the mapping from independent var idx to matrix index
  int nr_vals = N * (N - 1) / 2;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat2vec = Eigen::MatrixXd::Zero(N, N);
  int count = 0;
  for (size_t i = 0; i < N - 1; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      mat2vec(i, j) = count;
      count += 1;
    }
  }

  // creating the inverse map sparse matrix and reserve memory
  int nrNZ_per_row_off_diag = 2 * (N0 - 1) + 1;
  // int nrNZ_off_diag = nrNZ_per_row_off_diag * nr_vals;

  // resize the inverse matrix to the appropriate size
  // this won't reserve any memory for non-zero values
  A_inv->resize(nr_vals, nr_vals);

  // temporary vector storing column for holding the non zero entries
  std::vector<Eigen::Triplet<double>> temp_column;
  temp_column.reserve(nrNZ_per_row_off_diag);

  // for creating columns in inv_A
  for (size_t i = 0; i < N - 1; ++i) {
    TEASER_INFO_MSG_THROTTLE("Linear proj at i=" << i << "\n", i, 10);
    for (size_t j = i + 1; j < N; ++j) {
      // get current column index
      // var_j_idx is unique for all each loop, i.e., each var_j_idx only occurs once and the loops
      // won't enter the same column twice
      int var_j_idx = mat2vec(i, j);

      // start a inner vector
      // A_inv is column major so this will enable us to insert values to the end of column
      // var_j_idx
      A_inv->startVec(var_j_idx);

      // flag to indicated whether a diagonal entry on this column has been inserted
      // this is used to save computation time for adding x to all the diagonal entries
      bool diag_inserted = false;
      size_t diag_idx = 0;

      for (size_t p = 0; p < N; ++p) {
        if ((p != j) && (p != i)) {
          int var_i_idx;
          double entry_val;
          if (p < i) {
            // same row i, i,j upper triangular, i,p lower triangular
            // flip to upper-triangular
            var_i_idx = mat2vec(p, i);
            entry_val = y * theta_prepended(j) * theta_prepended(p);
          } else {
            var_i_idx = mat2vec(i, p);
            entry_val = -y * theta_prepended(j) * theta_prepended(p);
          }
          temp_column.emplace_back(var_i_idx, var_j_idx, entry_val);
          if (var_i_idx == var_j_idx) {
            diag_inserted = true;
            diag_idx = temp_column.size() - 1;
          }
        }
      }
      for (size_t p = 0; p < N; ++p) {
        if ((p != i) && (p != j)) {
          int var_i_idx;
          double entry_val;
          if (p < j) {
            // flip to upper-triangular
            var_i_idx = mat2vec(p, j);
            entry_val = -y * theta_prepended(i) * theta_prepended(p);
          } else {
            var_i_idx = mat2vec(j, p);
            entry_val = y * theta_prepended(i) * theta_prepended(p);
          }
          temp_column.emplace_back(var_i_idx, var_j_idx, entry_val);
          if (var_i_idx == var_j_idx) {
            diag_inserted = true;
            diag_idx = temp_column.size() - 1;
          }
        }
      }

      // insert diagonal entries if not already done so
      if (!diag_inserted) {
        temp_column.emplace_back(var_j_idx, var_j_idx, x);
      } else {
        double entry_val = temp_column[diag_idx].value() + x;
        temp_column[diag_idx] = {var_j_idx, var_j_idx, entry_val};
      }

      // sort by row index (ascending)
      std::sort(temp_column.begin(), temp_column.end(),
                [](const Eigen::Triplet<double>& t1, const Eigen::Triplet<double>& t2) {
                  return t1.row() < t2.row();
                });

      // populate A_inv with the temporary column
      for (size_t tidx = 0; tidx < temp_column.size(); ++tidx) {
        // take care of the diagonal entries
        A_inv->insertBack(temp_column[tidx].row(), var_j_idx) = temp_column[tidx].value();
      }
      temp_column.clear();
      temp_column.reserve(nrNZ_per_row_off_diag);
    }
  }
  TEASER_DEBUG_INFO_MSG("Finalizing A_inv ...");
  A_inv->finalize();
  TEASER_DEBUG_INFO_MSG("A_inv finalized.");
}

void teaser::DRSCertifier::getBlockRowSum(const Eigen::MatrixXd& A, const int& row,
                                          const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta,
                                          Eigen::Vector4d* output) {
  // unit = sparse(4,1); unit(end) = 1;
  // vector = kron(theta,unit); % vector of size 4N+4 by 1
  // entireRow = A(blkIndices(row,4),:); % entireRow of size 4 by 4N+4
  // row_sum_last_column = entireRow * vector; % last column sum has size 4 by 1;
  Eigen::Matrix<double, 4, 1> unit = Eigen::Matrix<double, 4, 1>::Zero();
  unit(3, 0) = 1;
  Eigen::Matrix<double, Eigen::Dynamic, 1> vector =
      vectorKron<double, Eigen::Dynamic, 4>(theta.transpose(), unit);
  *output = A.middleRows<4>(row) * vector;
}
