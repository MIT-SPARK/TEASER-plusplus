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
#include <cmath>
#include <random>

#include <Eigen/Eigenvalues>

#include "teaser/registration.h"
#include "test_utils.h"

TEST(RotationSolverTest, FGRRotation) {
  double ALLOWED_ROTATION_ERROR = 1e-5;
  // Problem 1: Identity
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, 10);
    for (size_t i = 0; i < src_points.cols(); ++i) {
      src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_points = src_points;

    // Set up FGR
    teaser::FastGlobalRegistrationSolver::Params params{1000, 0.0337, 1.4, 1e-3};
    teaser::FastGlobalRegistrationSolver fgr_solver(params);

    Eigen::Matrix3d result;
    fgr_solver.solveForRotation(src_points, dst_points, &result, nullptr);
    Eigen::Matrix3d ref_result;
    ref_result.setIdentity();
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_result << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << result << std::endl;

    EXPECT_TRUE((result - ref_result).norm() < ALLOWED_ROTATION_ERROR);
  }
  // Problem 2: Random rotation around x / y / z axis
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, 10);
    for (size_t i = 0; i < src_points.cols(); ++i) {
      src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
    }
    Eigen::Matrix3d ref_R;
    std::uniform_real_distribution<double> unif(0, 2 * M_PI);
    std::default_random_engine re;

    // Prepare solver
    teaser::FastGlobalRegistrationSolver::Params params{1000, 0.0337, 1.4, 1e-3};
    teaser::FastGlobalRegistrationSolver fgr_solver(params);
    Eigen::Matrix3d R;
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_points;

    // Rotation around x
    double theta = unif(re);
    // clang-format off
    ref_R << 1, 0,               0,
             0, std::cos(theta), -std::sin(theta),
             0, std::sin(theta), std::cos(theta);
    // clang-format on
    dst_points = ref_R * src_points;
    fgr_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);

    // Rotation around y
    // clang-format off
    ref_R << std::cos(theta), 0, std::sin(theta),
             0,               1, 0,
             -std::sin(theta),0, std::cos(theta);
    // clang-format on
    dst_points = ref_R * src_points;
    fgr_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);

    // Rotation around z
    ref_R << std::cos(theta), -std::sin(theta), 0, std::sin(theta), std::cos(theta), 0, 0, 0, 1;
    // clang-format on
    dst_points = ref_R * src_points;
    fgr_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);
  }
  // Problem 3: A more complex case
  {
    // Read in data
    std::ifstream source_file("./data/registration_test/rotation_only_src.csv");
    Eigen::Matrix<double, Eigen::Dynamic, 3> source_points =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, 3>(source_file);
    Eigen::Matrix<double, 3, Eigen::Dynamic> src = source_points.transpose();

    // Perform Arbitrary rotation
    Eigen::Matrix3d expected_R;
    // clang-format off
    expected_R << 0.997379773225804, -0.019905935977315, -0.069551000516966,
                  0.013777311189888, 0.996068297974922, -0.087510750572249,
                  0.071019530105605, 0.086323226782879, 0.993732623426126;
    // clang-format on
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst = expected_R * src;

    // Set up FGR
    // Since we have no noise, 1 iteration should give us the optimal solution.
    teaser::FastGlobalRegistrationSolver::Params params{1, 0.025, 1.4, 1e-3};
    teaser::FastGlobalRegistrationSolver fgr_solver(params);

    Eigen::Matrix3d result;
    fgr_solver.solveForRotation(src, dst, &result, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << expected_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << result << std::endl;

    EXPECT_TRUE(teaser::test::getAngularError(expected_R, result) < ALLOWED_ROTATION_ERROR);
  }
}

TEST(RotationSolverTest, GNCTLS) {
  double ALLOWED_ROTATION_ERROR = 1e-5;
  // Problem 1: Identity
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, 10);
    for (size_t i = 0; i < src_points.cols(); ++i) {
      src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
    }
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_points = src_points;

    // Set up GNC-TLS solver
    teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 1e-3};
    teaser::GNCTLSRotationSolver tls_solver(params);

    Eigen::Matrix3d result;
    tls_solver.solveForRotation(src_points, dst_points, &result, nullptr);
    Eigen::Matrix3d ref_result;
    ref_result.setIdentity();
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_result << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << result << std::endl;

    EXPECT_TRUE((result - ref_result).norm() < ALLOWED_ROTATION_ERROR);
  }
  // Problem 2: Random rotation around x / y / z axis
  {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points(3, 10);
    for (size_t i = 0; i < src_points.cols(); ++i) {
      src_points.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1);
    }
    Eigen::Matrix3d ref_R;
    std::uniform_real_distribution<double> unif(0, 2 * M_PI);
    std::default_random_engine re;

    // Set up GNC-TLS solver
    teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 1e-3};
    teaser::GNCTLSRotationSolver tls_solver(params);
    Eigen::Matrix3d R;
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst_points;

    // Rotation around x
    double theta = unif(re);
    // clang-format off
    ref_R << 1, 0,               0,
        0, std::cos(theta), -std::sin(theta),
        0, std::sin(theta), std::cos(theta);
    // clang-format on
    dst_points = ref_R * src_points;
    tls_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);

    // Rotation around y
    // clang-format off
    ref_R << std::cos(theta), 0, std::sin(theta),
        0,                    1, 0,
        -std::sin(theta),     0, std::cos(theta);
    // clang-format on
    dst_points = ref_R * src_points;
    tls_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);

    // Rotation around z
    // clang-format off
    ref_R << std::cos(theta), -std::sin(theta), 0,
             std::sin(theta), std::cos(theta),  0,
             0,               0,                1;
    // clang-format on
    dst_points = ref_R * src_points;
    tls_solver.solveForRotation(src_points, dst_points, &R, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << ref_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(ref_R, R) < ALLOWED_ROTATION_ERROR);
  }
  // Problem 3: A more complex case
  {
    // Read in data
    std::ifstream source_file("./data/registration_test/rotation_only_src.csv");
    Eigen::Matrix<double, Eigen::Dynamic, 3> source_points =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, 3>(source_file);
    Eigen::Matrix<double, 3, Eigen::Dynamic> src = source_points.transpose();

    // Perform Arbitrary rotation
    Eigen::Matrix3d expected_R;
    // clang-format off
    expected_R << 0.997379773225804, -0.019905935977315, -0.069551000516966,
                  0.013777311189888, 0.996068297974922, -0.087510750572249,
                  0.071019530105605, 0.086323226782879, 0.993732623426126;
    // clang-format on
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst = expected_R * src;

    // Set up TLS
    teaser::GNCTLSRotationSolver::Params params{100, 1e-12, 1.4, 1e-3};
    teaser::GNCTLSRotationSolver tls_solver(params);

    Eigen::Matrix3d result;
    tls_solver.solveForRotation(src, dst, &result, nullptr);
    std::cout << "Expected R: " << std::endl;
    std::cout << expected_R << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << result << std::endl;

    EXPECT_TRUE(teaser::test::getAngularError(expected_R, result) < ALLOWED_ROTATION_ERROR);
  }
}
