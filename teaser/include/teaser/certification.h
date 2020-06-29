/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace teaser {

/**
 * Abstract virtual class representing certification of registration results
 */
class AbstractRotationCertifier {
public:
  virtual ~AbstractRotationCertifier() {}

  /**
   * Abstract function for certifying rotation estimation results
   * @param rotation_solution [in] a solution to the rotatoin registration problem
   * @param src [in] Assume dst = R * src
   * @param dst [in] Assume dst = R * src
   * @param theta [in] a binary vector indicating inliers vs. outliers
   * @return  relative sub-optimality gap
   */
  virtual double certify(const Eigen::Matrix<double, 3, 3>& rotation_solution,
                         const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                         const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                         const Eigen::Matrix<bool, 1, Eigen::Dynamic>& theta) = 0;
};

/**
 * A rotation registration certifier using Douglas–Rachford Splitting (DRS)
 *
 * Please refer to:
 * [1] H. Yang, J. Shi, and L. Carlone, “TEASER: Fast and Certifiable Point Cloud Registration,”
 * arXiv:2001.07715 [cs, math], Jan. 2020.
 */
class DRSCertifier : public AbstractRotationCertifier {
public:
  DRSCertifier() = delete;

  /**
   * Constructor for DRSCertifier
   * @param noise_bound [in] bound on the noise
   * @param cbar2 [in] maximal allowed residual^2 to noise bound^2 ratio, usually set to 1
   */
  DRSCertifier(double noise_bound, double cbar2) : noise_bound_(noise_bound), cbar2_(cbar2){};

private:
  /**
   * Calculate the inverse of the linear projection matrix A mentioned in Theorem 35 of our TEASER
   * paper [1].
   *
   * @param theta [in] a binary vector indicating inliers vs. outliers, with 1 prepended
   * @param A_inv [out] inverse of A
   */
  void getLinearProjection(const Eigen::Matrix<bool, 1, Eigen::Dynamic>& theta,
                           Eigen::SparseMatrix<double>* A_inv);

  /**
   * Bounds on the noise for the measurements
   */
  double noise_bound_;

  /**
   * Maximal allowed residual^2 to noise bound^2 ratio, usually set to 1
   */
  double cbar2_;
};

} // namespace teaser