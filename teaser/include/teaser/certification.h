/**
 * Copyright (c) 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCore>

namespace teaser {

struct CertificationResult {
  double best_suboptimality = -1;
  std::vector<double> suboptimality_traj;
};

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
  virtual CertificationResult certify(const Eigen::Matrix3d& rotation_solution,
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

  /**
   * Main certification function
   *
   * @param R_solution [in] a feasible rotation solution
   * @param src [in] vectors under rotation
   * @param dst [in] vectors after rotation
   * @param theta [in] binary (1 vs. 0) vector indicating inliers vs. outliers
   * @return  relative sub-optimality gap
   */
  CertificationResult certify(const Eigen::Matrix3d& R_solution,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                              const Eigen::Matrix<bool, 1, Eigen::Dynamic>& theta) override;

  /**
   * Main certification function
   *
   * @param R_solution [in] a feasible rotation solution
   * @param src [in] vectors under rotation
   * @param dst [in] vectors after rotation
   * @param theta [in] binary (1 vs. -1) vector indicating inliers vs. outliers
   * @return  relative sub-optimality gap
   */
  CertificationResult certify(const Eigen::Matrix3d& R_solution,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                              const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                              const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta);

  /**
   * Compute sub-optimality gap
   * @param M
   * @param mu
   * @param N
   * @return
   */
  double computeSubOptimalityGap(const Eigen::MatrixXd& M, double mu, int N);

  /**
   * Get the Omega_1 matrix given a quaternion
   * @param q an Eigen quaternion
   * @param omega1 4-by-4 omega_1 matrix
   */
  Eigen::Matrix4d getOmega1(const Eigen::Quaterniond& q);

  /**
   * Get a 4-by-4 block diagonal matrix with each block represents omega_1
   * @param q
   * @param theta
   * @param D_omega
   */
  void getBlockDiagOmega(int Npm, const Eigen::Quaterniond& q,
                         Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* D_omega);

  /**
   * Get Q cost matrix (see Proposition 10 in [1])
   * @param v1 vectors under rotation
   * @param v2 vectors after rotation
   */
  void getQCost(const Eigen::Matrix<double, 3, Eigen::Dynamic>& v1,
                const Eigen::Matrix<double, 3, Eigen::Dynamic>& v2,
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* Q);

  /**
   * Given an arbitrary matrix W, project W to the correct dual structure
    (1) off-diagonal blocks must be skew-symmetric
    (2) diagonal blocks must satisfy W_00 = - sum(W_ii)
    (3) W_dual must also satisfy complementary slackness (because M_init satisfies complementary
   slackness) This projection is optimal in the sense of minimum Frobenious norm
   */
  void getOptimalDualProjection(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& W,
                                const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta_prepended,
                                const Eigen::SparseMatrix<double>& A_inv, Eigen::MatrixXd* W_dual);

  /**
   * Generate an initial guess (see Appendix U of [1]).
   *
   * The initial guess satisfies:
   * 1. KKT complementary slackness
   * 2. diagonal blocks of (Q - Lambda_guess) is PSD (except the first diagonal block)
   *
   * @param R [in] rotation matrix
   * @param theta [in] a binary (1 & -1) vector indicating inliers vs. outliers
   * @param src [in]
   * @param dst [in]
   * @param lambda_guess [out]
   */
  void getLambdaGuess(const Eigen::Matrix<double, 3, 3>& R,
                      const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic>& src,
                      const Eigen::Matrix<double, 3, Eigen::Dynamic>& dst,
                      Eigen::SparseMatrix<double>* lambda_guess);

  /**
   * Calculate the inverse of the linear projection matrix A mentioned in Theorem 35 of our TEASER
   * paper [1].
   *
   * @param theta_prepended [in] a binary (1 & -1) vector indicating inliers vs. outliers, with 1
   * prepended
   * @param A_inv [out] inverse of A
   */
  void getLinearProjection(const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta_prepended,
                           Eigen::SparseMatrix<double>* A_inv);

private:
  /**
   * Calculate the sum of a block (4-by-N) along the columns
   * @param A [in]
   * @param row [in]
   * @param theta [in]
   * @param output [out]
   */
  void getBlockRowSum(const Eigen::MatrixXd& A, const int& row,
                      const Eigen::Matrix<double, 1, Eigen::Dynamic>& theta,
                      Eigen::Vector4d* output);
  /**
   * Suboptimality gap
   *
   * This is not a percentage. Multiply by 100 to get a percentage.
   */
  double sub_optimality_ = 1e-3;

  /**
   * Maximum iterations allowed
   */
  double max_iterations_ = 2e2;

  /**
   * Gamma value (refer to [1] for details)
   */
  double gamma_ = 1.999999;

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