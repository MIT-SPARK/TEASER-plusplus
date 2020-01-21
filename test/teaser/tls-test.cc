/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

#include <Eigen/Eigenvalues>

#include "teaser/registration.h"
#include "test_utils.h"

TEST(TLSTest, TLSEstimate) {
  teaser::ScalarTLSEstimator tls;
  // No outlier
  {
    Eigen::RowVectorXd measurements(5);
    measurements << 0.5, 1, 0.6, 0.7, 1.2;
    Eigen::RowVectorXd ranges(5);
    ranges << 0.9, 0.9, 0.4, 0.5, 0.4;

    double ref_estimate = 0.8383;
    Eigen::Matrix<bool, 1, 5> ref_inliers;
    ref_inliers << true, true, true, true, true;

    double estimate_output;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers_output;
    inliers_output.resize(1, ref_inliers.cols());
    tls.estimate(measurements, ranges, &estimate_output, &inliers_output);
    EXPECT_NEAR(estimate_output, ref_estimate, 0.001); // TODO tolerance seems quite large

    for (size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(inliers_output(i), ref_inliers(i));
    }
  }
  // One outlier
  {
    Eigen::RowVectorXd measurements(6);
    measurements << 0.5, 1, 0.6, 0.7, 1.2, 10;
    Eigen::RowVectorXd ranges(6);
    ranges << 0.9, 0.9, 0.4, 0.5, 0.4, 0.5;

    double ref_estimate = 0.8383;
    Eigen::Matrix<bool, 1, 6> ref_inliers;
    ref_inliers << true, true, true, true, true, false;

    double estimate_output;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers_output;
    inliers_output.resize(1, ref_inliers.cols());
    tls.estimate(measurements, ranges, &estimate_output, &inliers_output);
    EXPECT_NEAR(estimate_output, ref_estimate, 0.001);

    for (size_t i = 0; i < 6; ++i) {
      EXPECT_EQ(inliers_output(i), ref_inliers(i));
    }
  }
  // Three (out of six) outliers
  {
    Eigen::RowVectorXd measurements(6);
    measurements << 0.5, 1, 0.6, 20, 16, 10;
    Eigen::RowVectorXd ranges(6);
    ranges << 0.9, 0.9, 0.4, 0.5, 0.4, 0.5;

    double ref_estimate = 0.6425;
    Eigen::Matrix<bool, 1, 6> ref_inliers;
    ref_inliers << true, true, true, false, false, false;

    double estimate_output;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers_output;
    inliers_output.resize(1, ref_inliers.cols());
    tls.estimate(measurements, ranges, &estimate_output, &inliers_output);
    EXPECT_NEAR(estimate_output, ref_estimate, 0.001);

    for (size_t i = 0; i < 6; ++i) {
      EXPECT_EQ(inliers_output(i), ref_inliers(i));
    }
  }
}

TEST(TLSTest, TLSEstimateTiled) {
  teaser::ScalarTLSEstimator tls;
  const int scale = 64;
  // No outlier
  {
    Eigen::RowVectorXd measurements(5);
    measurements << 0.5, 1, 0.6, 0.7, 1.2;
    Eigen::RowVectorXd ranges(5);
    ranges << 0.9, 0.9, 0.4, 0.5, 0.4;

    double ref_estimate = 0.8383;
    Eigen::Matrix<bool, 1, 5> ref_inliers;
    ref_inliers << true, true, true, true, true;

    double estimate_output;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers_output;
    inliers_output.resize(1, ranges.cols());
    tls.estimate_tiled(measurements, ranges, scale, &estimate_output, &inliers_output);
    EXPECT_NEAR(estimate_output, ref_estimate, 0.001); // TODO tolerance seems quite large

    for (size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(inliers_output(i), ref_inliers(i));
    }
  }
  // One outlier
  {
    Eigen::RowVectorXd measurements(6);
    measurements << 0.5, 1, 0.6, 0.7, 1.2, 10;
    Eigen::RowVectorXd ranges(6);
    ranges << 0.9, 0.9, 0.4, 0.5, 0.4, 0.5;

    double ref_estimate = 0.8383;
    Eigen::Matrix<bool, 1, 6> ref_inliers;
    ref_inliers << true, true, true, true, true, false;

    double estimate_output;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers_output;
    inliers_output.resize(1, ranges.cols());
    tls.estimate_tiled(measurements, ranges, scale, &estimate_output, &inliers_output);
    EXPECT_NEAR(estimate_output, ref_estimate, 0.001);

    for (size_t i = 0; i < 6; ++i) {
      EXPECT_EQ(inliers_output(i), ref_inliers(i));
    }
  }
  // Three (out of six) outliers
  {
    Eigen::RowVectorXd measurements(6);
    measurements << 0.5, 1, 0.6, 20, 16, 10;
    Eigen::RowVectorXd ranges(6);
    ranges << 0.9, 0.9, 0.4, 0.5, 0.4, 0.5;

    double ref_estimate = 0.6425;
    Eigen::Matrix<bool, 1, 6> ref_inliers;
    ref_inliers << true, true, true, false, false, false;

    double estimate_output;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> inliers_output;
    inliers_output.resize(1, ranges.cols());
    tls.estimate_tiled(measurements, ranges, scale, &estimate_output, &inliers_output);
    EXPECT_NEAR(estimate_output, ref_estimate, 0.001);

    for (size_t i = 0; i < 6; ++i) {
      EXPECT_EQ(inliers_output(i), ref_inliers(i));
    }
  }
}
