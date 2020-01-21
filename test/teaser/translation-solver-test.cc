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

TEST(TranslationTest, TLSTranslation) {
  // Problem 1: Zero translation
  {
    std::ifstream objectFile("./data/registration_test/translation_test_v1_inliers.csv");
    auto object_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(objectFile);

    // Parameters for estimating translation
    double noise_bound = 0.025; // arbitrary
    double cbar2 = 1;

    // Estimating translation
    Eigen::Vector3d actual_t;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> actual_inliers;
    actual_inliers.resize(1, object_points.cols());
    teaser::TLSTranslationSolver translationSolver(noise_bound, cbar2);
    // Pass the same vector of points b/c we are testing for zero translation
    translationSolver.solveForTranslation(object_points, object_points, &actual_t, &actual_inliers);

    // Expected
    Eigen::Vector3d expected_t = Eigen::Vector3d::Zero();
    EXPECT_TRUE((actual_t - expected_t).norm() < 1e-5);
  }
  // Problem 2: Translation in x / y / z only
  {
    std::ifstream objectFile("./data/registration_test/translation_test_v1_inliers.csv");
    auto object_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(objectFile);

    // Parameters for estimating translation
    double noise_bound = 0.025; // arbitrary
    double cbar2 = 1;

    // Estimating translation
    // 1. in x only
    Eigen::Matrix<double, 3, Eigen::Dynamic> translated_points = object_points;
    translated_points.row(0).array() += 1;
    Eigen::Vector3d actual_t;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> actual_inliers;
    actual_inliers.resize(1, object_points.cols());
    teaser::TLSTranslationSolver translationSolver(noise_bound, cbar2);
    translationSolver.solveForTranslation(object_points, translated_points, &actual_t,
                                          &actual_inliers);

    // Expected
    Eigen::Vector3d expected_t;
    expected_t << 1, 0, 0;
    EXPECT_TRUE((actual_t - expected_t).norm() < 1e-5);

    // 2. in y only
    translated_points = object_points;
    translated_points.row(1).array() += 1;
    translationSolver.solveForTranslation(object_points, translated_points, &actual_t,
                                          &actual_inliers);

    // Expected
    expected_t << 0, 1, 0;
    EXPECT_TRUE((actual_t - expected_t).norm() < 1e-5);

    // 3. in z only
    translated_points = object_points;
    translated_points.row(2).array() += 1;
    translationSolver.solveForTranslation(object_points, translated_points, &actual_t,
                                          &actual_inliers);

    // Expected
    expected_t << 0, 0, 1;
    EXPECT_TRUE((actual_t - expected_t).norm() < 1e-5);
  }
  // Problem 3: An arbitrary translation
  {
    // Prepare input data
    std::ifstream objectFile("./data/registration_test/translation_test_v1_inliers.csv");
    std::ifstream sceneFile("./data/registration_test/translation_test_v2_inliers.csv");
    auto object_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(objectFile);
    auto scene_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(sceneFile);
    EXPECT_EQ(object_points.cols(), scene_points.cols());

    // Parameters for estimating translation
    double noise_bound = 0.00673642835;
    double cbar2 = 1;

    // Estimating translation
    Eigen::Vector3d actual_t;
    Eigen::Matrix<bool, 1, Eigen::Dynamic> actual_inliers;
    actual_inliers.resize(1, object_points.cols());
    teaser::TLSTranslationSolver translationSolver(noise_bound, cbar2);
    translationSolver.solveForTranslation(object_points, scene_points, &actual_t, &actual_inliers);

    // Expected
    Eigen::Vector3d expected_t;
    expected_t << -0.098430131086161, 0.008679113091532, 0.197317864174211;
    EXPECT_TRUE((actual_t - expected_t).norm() < 1e-5);
  }
}
