/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "teaser/certification.h"
#include "teaser/geometry.h"

void teaser::DRSCertifier::getLambdaGuess(const Eigen::Matrix<double, 3, 3>& R,
                                          const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta,
                                          const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                                          const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                                          Eigen::SparseMatrix<double>* lambda_guess) {
  int K = theta.size();
  int Npm = 4 * K + 4;

  // prepare the lambda sparse matrix output
  lambda_guess->resize(Npm, Npm);
  lambda_guess->reserve(Npm * (Npm-1) * 2);
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
      current_block(3, 3) = -0.75 * xi.squaredNorm() - 0.25 * cbar2;

      // compute the top-left 3-by-3 block
      current_block.topLeftCorner<3, 3>() =
          src_i_hatmap * src_i_hatmap -
          0.5 * (src.col(i).transpose() * xi) * Eigen::Matrix3d::Identity() +
          0.5 * xi_hatmap * src_i_hatmap + 0.5 * xi * src.col(i).transpose() -
          0.75 * xi.squaredNorm() * Eigen::Matrix3d::Identity() -
          0.25 * cbar2 * Eigen::Matrix3d::Identity();

      // compute the vector part
      current_block.topLeftCorner<3, 1>() = -1.5 * xi_hatmap * src.col(i);
      current_block.bottomLeftCorner<1, 3>() = (current_block.topLeftCorner<3, 1>()).transpose();
    } else {
      // residual
      Eigen::Matrix<double, 3, 1> phi = R.transpose() * (dst.col(i) - R * src.col(i));
      Eigen::Matrix<double, 3, 3> phi_hatmap = teaser::hatmap(phi);

      // compute lambda_i, (4,4) entry
      current_block(3, 3) = -0.25 * phi.squaredNorm() - 0.75 * cbar2;

      // compute E_ii, top-left 3-by-3 block
      current_block.topLeftCorner<3, 3>() =
          src_i_hatmap * src_i_hatmap -
          0.5 * (src.col(i).transpose() * phi) * Eigen::Matrix3d::Identity() +
          0.5 * phi_hatmap * src_i_hatmap + 0.5 * phi * src.col(i) -
          0.25 * phi.squaredNorm() * Eigen::Matrix3d::Identity() -
          0.25 * cbar2 * Eigen::Matrix3d::Identity();

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
