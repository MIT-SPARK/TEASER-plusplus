/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "teaser/certification.h"
#include "teaser/linalg.h"

void teaser::DRSCertifier::getOptimalDualProjection(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& W,
    const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta_prepended,
    const Eigen::SparseMatrix<double>& A_inv,
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
      int col_idx_end = i * 4 + 3;

      // current theta value calculation
      double theta_ij = theta_prepended.col(i) * theta_prepended.col(j);

      // [-theta_ij 1]
      Eigen::Matrix<double, 1, 2> temp_A;
      temp_A << -theta_ij, 1;

      // [-1 theta_ij]
      Eigen::Matrix<double, 1, 2> temp_B;
      temp_A << -1, theta_ij;

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
  W_dual->setZero();
  W_dual->resize(Npm, Npm);
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

    for (size_t j = i+1; j < N+1; ++j) {
      int col_idx_start = j*4;

      // take W_ij and break into top-left 3x3 and vectors
      W_ij = W_i.block(0, col_idx_start, 4,4);
      y_dual_ij = (b_W_dual.row(count)).transpose();

      // assemble W_dual_ij
      W_dual_ij = (W_ij - W_ij.transpose())/2;
      W_dual_ij.block<3,1>(0,3) = y_dual_ij;
      W_dual_ij.block<1,3>(3, 0) = -y_dual_ij.transpose();

      // assign W_dual_ij to W_dual_i
      W_dual_i.block<4, 4>(0, col_idx_start) = W_dual_ij;

      count +=1;
    }
    W_dual->block(row_idx_start, 0, 4, Npm) = W_dual_i;
  }
  *W_dual = *W_dual + W_dual->transpose();

  // Project the diagonal blocks
  Eigen::Matrix4d W_diag_mean = Eigen::Matrix4d::Zero();
  Eigen::Matrix3d W_diag_sum_33 = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < N+1; ++i) {

  }
  //for i = 1:N+1
  //    idx = blkIdxGroups{i};
  //    W_dual_row_sum_last_column = get_block_row_sum_last_column(W_dual,i,theta);
  //    W_ii = W(idx,idx);
  //    % modify W_ii's last column/row to satisfy complementary slackness
  //    W_ii(:,4) = -theta(i) * W_dual_row_sum_last_column;
  //    W_ii(4,:) = -theta(i) * W_dual_row_sum_last_column';
  //
  //    W_dual(idx,idx) = W_ii;
  //
  //    W_diag_sum_33 = W_diag_sum_33 + W_ii(1:3,1:3);
  //end
  //W_diag_mean(1:3,1:3) = W_diag_sum_33 / (N+1);
  //W_dual = W_dual - kron(speye(N+1),W_diag_mean);
}

void teaser::DRSCertifier::getLambdaGuess(const Eigen::Matrix<double, 3, 3>& R,
                                          const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta,
                                          const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                                          const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                                          Eigen::SparseMatrix<double>* lambda_guess) {
  int K = theta.size();
  int Npm = 4 * K + 4;

  // prepare the lambda sparse matrix output
  lambda_guess->resize(Npm, Npm);
  lambda_guess->reserve(Npm * (Npm - 1) * 2);
  lambda_guess->setZero();

  // 4-by-4 Eigen matrix to store the top left 4-by-4 block
  Eigen::Matrix<double, 4, 4> topleft_block = Eigen::Matrix4d::Zero();

  // 4-by-4 Eigen matrix to store the current 4-by-4 block
  Eigen::Matrix<double, 4, 4> current_block = Eigen::Matrix4d::Zero();

  // Eigen triplets vector for sparse matrix construction
  std::vector<Eigen::Triplet<double>> sparse_triplets;

  for (size_t i = 0; i < K; ++i) {
    // hat maps for later usage
    Eigen::Matrix<double, 3, 3> src_i_hatmap = teaser::hatmap(src.col(i));
    if (theta(1, i) > 0) {
      // residual
      Eigen::Matrix<double, 3, 1> xi = R.transpose() * (dst.col(i) - R * src.col(i));
      Eigen::Matrix<double, 3, 3> xi_hatmap = teaser::hatmap(xi);

      // compute the (4,4) entry of the current block, obtained from KKT complementary slackness
      current_block(3, 3) = -0.75 * xi.squaredNorm() - 0.25 * cbar2_;

      // compute the top-left 3-by-3 block
      current_block.topLeftCorner<3, 3>() =
          src_i_hatmap * src_i_hatmap -
          0.5 * (src.col(i)).dot(xi) * Eigen::Matrix3d::Identity() +
          0.5 * xi_hatmap * src_i_hatmap + 0.5 * xi * src.col(i).transpose() -
          0.75 * xi.squaredNorm() * Eigen::Matrix3d::Identity() -
          0.25 * cbar2_ * Eigen::Matrix3d::Identity();

      // compute the vector part
      current_block.topLeftCorner<3, 1>() = -1.5 * xi_hatmap * src.col(i);
      current_block.bottomLeftCorner<1, 3>() = (current_block.topLeftCorner<3, 1>()).transpose();
    } else {
      // residual
      Eigen::Matrix<double, 3, 1> phi = R.transpose() * (dst.col(i) - R * src.col(i));
      Eigen::Matrix<double, 3, 3> phi_hatmap = teaser::hatmap(phi);

      // compute lambda_i, (4,4) entry
      current_block(3, 3) = -0.25 * phi.squaredNorm() - 0.75 * cbar2_;

      // compute E_ii, top-left 3-by-3 block
      current_block.topLeftCorner<3, 3>() =
          src_i_hatmap * src_i_hatmap -
          0.5 * (src.col(i)).dot(phi) * Eigen::Matrix3d::Identity() +
          0.5 * phi_hatmap * src_i_hatmap + 0.5 * phi * src.col(i).transpose() -
          0.25 * phi.squaredNorm() * Eigen::Matrix3d::Identity() -
          0.25 * cbar2_ * Eigen::Matrix3d::Identity();

      // compute x_i
      current_block.topLeftCorner<3, 1>() = -0.5 * phi_hatmap * src.col(i);
      current_block.bottomLeftCorner<1, 3>() = (current_block.topLeftCorner<3, 1>()).transpose();
    }

    // put the current block to the sparse triplets
    // start idx: i * 4
    // end idx: i * 4 + 3
    // assume current block is column major
    for (size_t col = 0; col < 4; ++col) {
      for (size_t row = 0; row < 4; ++row) {
        sparse_triplets.emplace_back(i * 4 + row, i * 4 + col, -current_block(row, col));
      }
    }

    // update the first block
    topleft_block += current_block;
  }

  // put the first block to the sparse triplets
  for (size_t col = 0; col < 4; ++col) {
    for (size_t row = 0; row < 4; ++row) {
      sparse_triplets.emplace_back(row, col, topleft_block(row, col));
    }
  }

  // construct the guess as a sparse matrix
  lambda_guess->setFromTriplets(sparse_triplets.begin(), sparse_triplets.end());
}

void teaser::DRSCertifier::getLinearProjection(
    const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta_prepended,
    Eigen::SparseMatrix<double>* A_inv) {
  // number of off-diagonal entries in the inverse map
  int N0 = theta_prepended.cols() - 1;

  int y = 1 / (2 * N0 + 6);
  // number of diagonal entries in the inverse map
  int x = (N0 + 1) * y;

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
  int nrNZ_per_row_off_diag = 2 * (N0 - 1);
  int nrNZ_off_diag = nrNZ_per_row_off_diag * nr_vals;
  A_inv->resize(nr_vals, nr_vals);
  A_inv->setZero();

  // for creating columns in inv_A
  std::vector<Eigen::Triplet<double>> sparse_triplets;
  for (size_t i = 0; i < N - 1; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      int var_1_idx = mat2vec(i, j);

      for (size_t p = 0; p < N; ++p) {
        if ((p != j) && (p != i)) {
          int var_2_idx;
          double entry_val;
          if (p < i) {
            // same row i, i,j upper triangular, i,p lower triangular
            // flip to upper-triangular
            var_2_idx = mat2vec(p, i);
            entry_val = y * theta_prepended(j) * theta_prepended(p);
          } else {
            var_2_idx = mat2vec(i, p);
            entry_val = -y * theta_prepended(j) * theta_prepended(p);
          }
          sparse_triplets.emplace_back(var_2_idx, var_1_idx, entry_val);
        }
      }
      for (size_t p = 0; p < N; ++p) {
        if ((p != i) && (p != j)) {
          int var_2_idx;
          double entry_val;
          if (p < j) {
            // flip to upper-triangular
            var_2_idx = mat2vec(p, j);
            entry_val = -y * theta_prepended(i) * theta_prepended(p);
          } else {
            var_2_idx = mat2vec(j, p);
            entry_val = y * theta_prepended(i) * theta_prepended(p);
          }
          sparse_triplets.emplace_back(var_2_idx, var_1_idx, entry_val);
        }
      }
    }
  }
  // create diagonal entries
  for (size_t i = 0; i < nr_vals; ++i) {
    sparse_triplets.emplace_back(i, i, x);
  }
  A_inv->setFromTriplets(sparse_triplets.begin(), sparse_triplets.end());
}
