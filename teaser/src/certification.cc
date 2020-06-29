/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "teaser/certification.h"

void teaser::DRSCertifier::getLinearProjection(const Eigen::Matrix<bool, 1, Eigen::Dynamic>& theta,
                                               Eigen::SparseMatrix<double>* A_inv) {
  // number of off-diagonal entries in the inverse map
  int N0 = theta.cols() - 1;

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
            entry_val = y * theta(j) * theta(p);
          } else {
            var_2_idx = mat2vec(i, p);
            entry_val = -y * theta(j) * theta(p);
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
            entry_val = -y * theta(i) * theta(p);
          } else {
            var_2_idx = mat2vec(j, p);
            entry_val = y * theta(i) * theta(p);
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
