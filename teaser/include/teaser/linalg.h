/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

namespace teaser {

/**
 * Vector-vector kronecker product function
 * @tparam NumT
 * @tparam N size of the first vector
 * @tparam M size of the second vector
 * @param v1 first vector
 * @param v2 second vector
 */
template <typename NumT, int N, int M>
void vectorKron(const Eigen::Matrix<NumT, N, 1>& v1, const Eigen::Matrix<NumT, M, 1>& v2,
                Eigen::Matrix<NumT, N * M, 1>* output) {
#pragma omp parallel for collapse(2) shared(v1, v2, output)
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      (*output)[i*N+j] = v1[i] * v2[j];
    }
  }
}

} // namespace teaser