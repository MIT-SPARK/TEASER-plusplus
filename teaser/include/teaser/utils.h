/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

namespace teaser {
namespace utils {

/**
 * A templated random sample function (w/o replacement). Based on MATLAB implementation of
 * randsample()
 * @tparam T A number type
 * @tparam URBG A UniformRandomBitGenerator type
 * @param input An input vector containing the whole population
 * @param num_samples Number of samples we want
 * @param g
 * @return
 */
template <class T, class URBG>
std::vector<T> randomSample(std::vector<T> input, size_t num_samples, URBG&& g) {

  std::vector<T> output;
  output.reserve(num_samples);
  if (4 * num_samples > input.size()) {
    // if the sample is a sizeable fraction of the whole population,
    // just randomly shuffle the entire population and return the
    // first num_samples
    std::shuffle(input.begin(), input.end(), g);
    for (size_t i = 0; i < num_samples; ++i) {
      output.push_back(input[i]);
    }
  } else {
    // if the sample is small, repeatedly sample with replacement until num_samples
    // unique values
    std::unordered_set<size_t> sample_indices;
    std::uniform_int_distribution<> dis(0, input.size());
    while (sample_indices.size() < num_samples) {
      sample_indices.insert(dis(std::forward<URBG>(g)));
    }
    for (auto&& i : sample_indices) {
      output.push_back(input[i]);
    }
  }
  return output;
}

/**
 * Remove one row from a matrix.
 * Credit to: https://stackoverflow.com/questions/13290395
 * @param matrix an Eigen::Matrix.
 * @param rowToRemove index of row to remove. If >= matrix.rows(), no operation will be taken
 */
template <class T, int R, int C>
void removeRow(Eigen::Matrix<T, R, C>& matrix, unsigned int rowToRemove) {
  if (rowToRemove >= matrix.rows()) {
    return;
  }
  unsigned int numRows = matrix.rows() - 1;
  unsigned int numCols = matrix.cols();

  if (rowToRemove < numRows) {
    matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) =
        matrix.bottomRows(numRows - rowToRemove);
  }

  matrix.conservativeResize(numRows, numCols);
}

/**
 * Remove one column from a matrix.
 * Credit to: https://stackoverflow.com/questions/13290395
 * @param matrix
 * @param colToRemove index of col to remove. If >= matrix.cols(), no operation will be taken
 */
template <class T, int R, int C>
void removeColumn(Eigen::Matrix<T, R, C>& matrix, unsigned int colToRemove) {
  if (colToRemove >= matrix.cols()) {
    return;
  }
  unsigned int numRows = matrix.rows();
  unsigned int numCols = matrix.cols() - 1;

  if (colToRemove < numCols) {
    matrix.block(0, colToRemove, numRows, numCols - colToRemove) =
        matrix.rightCols(numCols - colToRemove);
  }

  matrix.conservativeResize(numRows, numCols);
}

/**
 * Helper function to calculate the diameter of a row vector of points
 * @param X
 * @return the diameter of the set of points given
 */
template <class T, int D> float calculateDiameter(const Eigen::Matrix<T, D, Eigen::Dynamic>& X) {
  Eigen::Matrix<T, D, 1> cog = X.rowwise().sum() / X.cols();
  Eigen::Matrix<T, D, Eigen::Dynamic> P = X.colwise() - cog;
  Eigen::Matrix<T, 1, Eigen::Dynamic> temp = P.array().square().colwise().sum();
  return 2 * std::sqrt(temp.maxCoeff());
}

/**
 * Helper function to use SVD to estimate rotation.
 * Method described here: http://igl.ethz.ch/projects/ARAP/svd_rot.pdf
 * @param X
 * @param Y
 * @return a rotation matrix R
 */
inline Eigen::Matrix3d svdRot(const Eigen::Matrix<double, 3, Eigen::Dynamic>& X,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& Y,
                              const Eigen::Matrix<double, 1, Eigen::Dynamic>& W) {
  // Assemble the correlation matrix H = X * Y'
  Eigen::Matrix3d H = X * W.asDiagonal() * Y.transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  if (U.determinant() * V.determinant() < 0) {
    V.col(2) *= -1;
  }

  return V * U.transpose();
}

/**
 * Modified helper function to use svd to estimate SO(2) rotation.
 * Method described here: http://igl.ethz.ch/projects/ARAP/svd_rot.pdf
 * @param X
 * @param Y
 * @return a rotation matrix R whose dimension is 2D
 */
inline Eigen::Matrix2d svdRot2d(const Eigen::Matrix<double, 2, Eigen::Dynamic>& X,
                                const Eigen::Matrix<double, 2, Eigen::Dynamic>& Y,
                                const Eigen::Matrix<double, 1, Eigen::Dynamic>& W) {
  // Assemble the correlation matrix H = X * Y'
  Eigen::Matrix2d H = X * W.asDiagonal() * Y.transpose();

  Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d U = svd.matrixU();
  Eigen::Matrix2d V = svd.matrixV();

  if (U.determinant() * V.determinant() < 0) {
    V.col(1) *= -1;
  }

  return V * U.transpose();
}

/**
 * Use a boolean Eigen matrix to mask a vector
 * @param mask a 1-by-N boolean Eigen matrix
 * @param elements vector to be masked
 * @return
 */
template <class T>
inline std::vector<T> maskVector(Eigen::Matrix<bool, 1, Eigen::Dynamic> mask,
                                 const std::vector<T>& elements) {
  assert(mask.cols() == elements.size());
  std::vector<T> result;
  for (size_t i = 0; i < mask.cols(); ++i) {
    if (mask(i)) {
      result.emplace_back(elements[i]);
    }
  }
  return result;
}

/**
 * Get indices of non-zero elements in an Eigen row vector
 * @param mask a 1-by-N boolean Eigen matrix
 * @return A vector containing indices of the true elements in the row vector
 */
template <class T>
inline std::vector<int> findNonzero(const Eigen::Matrix<T, 1, Eigen::Dynamic>& mask) {
  std::vector<int> result;
  for (size_t i = 0; i < mask.cols(); ++i) {
    if (mask(i)) {
      result.push_back(i);
    }
  }
  return result;
}

} // namespace utils
} // namespace teaser
