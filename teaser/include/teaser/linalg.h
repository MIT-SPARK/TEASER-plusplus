/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>

namespace teaser {

/**
 * Return the hat map of the provided vector (a skew symmetric matrix).
 * @param u 3-by-1 vector
 * @param x 3-by-3 skew symmetric matrix
 */
Eigen::Matrix<double, 3, 3> hatmap(const Eigen::Matrix<double, 3, 1>& u) {
  Eigen::Matrix<double, 3, 3> x;
  // clang-format off
  x << 0,           -u(2),  u(1),
      u(2),   0,          -u(0),
      -u(1),  u(0),  0;
  // clang-format on
  return x;
}

/**
 * Vector-vector kronecker product function with fixed-size output
 * @tparam NumT
 * @tparam N size of the first vector
 * @tparam M size of the second vector
 * @param v1 [in] first vector
 * @param v2 [in] second vector
 * @param output [out] output vector
 */
template <typename NumT, int N, int M>
void vectorKron(const Eigen::Matrix<NumT, N, 1>& v1, const Eigen::Matrix<NumT, M, 1>& v2,
                Eigen::Matrix<NumT, N * M, 1>* output) {
#pragma omp parallel for collapse(2) shared(v1, v2, output) default(none)
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      (*output)[i * M + j] = v1[i] * v2[j];
    }
  }
}

/**
 * Vector-vector kronecker product function with dynamic-size output
 * @tparam NumT numerical type for Eigen matrices (double, float, etc.)
 * @param v1 [in] first vector
 * @param v2 [in] second vector
 * @return Result of kronecker product
 */
template <typename NumT, int N, int M>
Eigen::Matrix<NumT, Eigen::Dynamic, 1> vectorKron(const Eigen::Matrix<NumT, N, 1>& v1,
                                                  const Eigen::Matrix<NumT, M, 1>& v2) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> output(v1.rows() * v2.rows(), 1);
#pragma omp parallel for collapse(2) shared(v1, v2, output) default(none)
  for (size_t i = 0; i < v1.rows(); ++i) {
    for (size_t j = 0; j < v2.rows(); ++j) {
      output[i * v2.rows() + j] = v1[i] * v2[j];
    }
  }
  return output;
}

/**
 * Find the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
 *
 * See: https://www.sciencedirect.com/science/article/pii/0024379588902236
 *
 * @tparam NumT numerical type for Eigen matrices (double, float, etc.)
 * @param A [in] input matrix
 * @param nearestPSD [out] output neaest positive semi-definite matrix
 * @param eig_threshold [in] optional threshold of determining the smallest eigen values
 */
template <typename NumT>
void getNearestPSD(const Eigen::Matrix<NumT, Eigen::Dynamic, Eigen::Dynamic>& A,
                   Eigen::Matrix<NumT, Eigen::Dynamic, Eigen::Dynamic>* nearestPSD) {
  assert(A.rows() == A.cols());
  nearestPSD->resize(A.rows(), A.cols());

  // symmetrize A into B
  Eigen::MatrixXd B = (A + A.transpose()) / 2;

  // eigendecomposition of B
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_B(B);
  Eigen::VectorXd De = eig_B.eigenvalues();
  Eigen::MatrixXd De_positive = (De.array() < 0).select(0, De).asDiagonal();
  Eigen::MatrixXd Ve = eig_B.eigenvectors();
  *nearestPSD = Ve * De_positive * Ve.transpose();
}

} // namespace teaser