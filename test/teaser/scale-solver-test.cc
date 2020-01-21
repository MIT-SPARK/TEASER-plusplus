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
#include <random>

#include <Eigen/Eigenvalues>

#include "teaser/registration.h"
#include "teaser/macros.h"
#include "test_utils.h"

TEST(ScaleSolverTest, UnknownScale) {
  double ACCEPTABLE_ERROR = 1e-5;

  // Read in data
  std::ifstream objectFile("./data/registration_test/objectIn.csv");
  auto object_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(objectFile);

  // Problem 1: No scaling
  {
    // Prepare parameters & solver
    double noise_bound = 1; // arbitrary
    int cbar2 = 1;          // arbitrary
    teaser::TLSScaleSolver solver(noise_bound, cbar2);

    // Solve for scale
    double actual_scale = 0;
    double expected_scale = 1;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> actual_inliers;
    actual_inliers.resize(1, object_points.cols());
    solver.solveForScale(object_points, object_points, &actual_scale, &actual_inliers);

    // Compare with expected values
    EXPECT_NEAR(expected_scale, actual_scale, ACCEPTABLE_ERROR);
  }
  // Problem 2: Random scaling
  {
    // Prepare parameters & solver
    double noise_bound = 1; // arbitrary
    int cbar2 = 1;          // arbitrary
    teaser::TLSScaleSolver solver(noise_bound, cbar2);

    // Scaling input points by a random value
    std::uniform_real_distribution<double> unif(0, 5);
    std::default_random_engine re;
    double expected_scale = unif(re);
    Eigen::Matrix<double, 3, Eigen::Dynamic> scaled_points = object_points.array() * expected_scale;

    // Solve for scale
    double actual_scale = 0;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> actual_inliers;
    actual_inliers.resize(1, object_points.cols());
    solver.solveForScale(object_points, scaled_points, &actual_scale, &actual_inliers);

    // Compare with expected values
    EXPECT_NEAR(expected_scale, actual_scale, ACCEPTABLE_ERROR);
  }
}

TEST(ScaleSolverTest, FixedScale) {
  // Read in data
  std::ifstream objectFile("./data/registration_test/objectIn.csv");
  auto object_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(objectFile);

  // Problem 1: No outliers
  {
    double noise_bound = 1; // arbitrary
    int cbar2 = 1;          // arbitrary
    teaser::ScaleInliersSelector solver(noise_bound, cbar2);

    double actual_scale = 0;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> actual_inliers;
    actual_inliers.resize(1, object_points.cols());
    solver.solveForScale(object_points, object_points, &actual_scale, &actual_inliers);

    EXPECT_EQ(actual_scale, 1);
    for (size_t i = 0; i < actual_inliers.cols(); ++i) {
      EXPECT_TRUE(actual_inliers(0, i));
    }
  }
  // Problem 2: All outliers
  {
    double noise_bound = 1; // arbitrary
    int cbar2 = 1;          // arbitrary
    // shift & scale the points so all points will be outliers
    Eigen::Matrix<double, 3, Eigen::Dynamic> shifted_object = object_points.array() * 3 + 10;
    teaser::ScaleInliersSelector solver(noise_bound, cbar2);

    double actual_scale = 0;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> actual_inliers;
    actual_inliers.resize(1, object_points.cols());
    solver.solveForScale(object_points, shifted_object, &actual_scale, &actual_inliers);

    EXPECT_EQ(actual_scale, 1);
    for (size_t i = 0; i < actual_inliers.cols(); ++i) {
      EXPECT_FALSE(actual_inliers(0, i));
    }
  }
  // Problem 3: One outlier
  {
    double noise_bound = 1; // arbitrary
    int cbar2 = 1;          // arbitrary
    // shift & scale the points so all points will be outliers
    Eigen::Matrix<double, 3, Eigen::Dynamic> shifted_object = object_points.array();
    shifted_object.col(0).array() *= 10;
    teaser::ScaleInliersSelector solver(noise_bound, cbar2);

    double actual_scale = 0;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> actual_inliers;
    actual_inliers.resize(1, object_points.cols());
    solver.solveForScale(object_points, shifted_object, &actual_scale, &actual_inliers);

    EXPECT_EQ(actual_scale, 1);
    EXPECT_FALSE(actual_inliers(0, 0));
    for (size_t i = 1; i < actual_inliers.cols(); ++i) {
      EXPECT_TRUE(actual_inliers(0, i));
    }
  }
}
