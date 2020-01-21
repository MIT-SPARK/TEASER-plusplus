/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <map>

#include "mex.h"

#include <Eigen/Core>

// Credit to Effective Modern C++ Item 10
template <typename E> constexpr typename std::underlying_type<E>::type toUType(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
};

/**
 * Templated function to check if input is a R-by-C MATLAB matrix
 * @tparam R rows
 * @tparam C columns
 * @param pa
 * @return
 */
template <int R, int C> bool isRealDoubleMatrix(const mxArray* pa) {
  bool isDoubleMatrix = mxIsDouble(pa) && (!mxIsComplex(pa)) && (!mxIsScalar(pa));
  bool correctDimensions = (mxGetM(pa) == R) && (mxGetN(pa) == C);
  return isDoubleMatrix && correctDimensions;
}

/**
 * Return true if input is a 3-by-N MATLAB matrix
 * @param pa
 * @return
 */
bool isPointCloudMatrix(const mxArray* pa) {
  size_t rows = 3;
  bool isDoubleMatrix = mxIsDouble(pa) && (!mxIsComplex(pa)) && (!mxIsScalar(pa));
  bool correctDimensions = mxGetM(pa) == rows;
  return isDoubleMatrix && correctDimensions;
}

/**
 * Return true if input is a real double scalar
 * @param pa
 * @return
 */
bool isRealDoubleScalar(const mxArray* pa) {
  return mxIsDouble(pa) && mxIsScalar(pa) && (!mxIsComplex(pa));
}

/**
 * Convert a 3-by-N mxArray to Eigen 3-by-N matrix
 * @param pa
 */
void mexPointMatrixToEigenMatrix(const mxArray* pa,
                                 Eigen::Matrix<double, 3, Eigen::Dynamic>* eigen_matrix) {
  int rows = mxGetM(pa);
  int cols = mxGetN(pa);
  mexPrintf("row: %d cols: %d \n", rows, cols);
  if (rows != 3)
    return;
  eigen_matrix->resize(rows, cols);

  double* in_matrix;
  in_matrix = mxGetPr(pa);

  *eigen_matrix = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(in_matrix, rows, cols);
}
