/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Core>

#include "teaser/certification.h"
#include "test_utils.h"

TEST(DRSCertifierTest, GetOmega1) {
  {
    // Case 1: N=10
    // read in expected data
    std::ifstream omega_source_file("./data/certification_test/case_1/omega.csv");
    std::ifstream q_source_file("./data/certification_test/case_1/q.csv");
    auto q_mat = teaser::test::readFileToEigenFixedMatrix<double, 4, 1>(q_source_file);
    auto expected_output =
        teaser::test::readFileToEigenFixedMatrix<double, 4, 4>(omega_source_file);
    double cbar2 = 1;
    double noise_bound = 4.594291399787397e-02;

    // construct the certifier
    teaser::DRSCertifier certifier(noise_bound, cbar2);
    Eigen::Quaternion<double> q(q_mat(0), q_mat(1), q_mat(2), q_mat(3));

    // perform the computation
    Eigen::Matrix4d actual_output = certifier.getOmega1(q);
    if (!actual_output.isApprox(expected_output)) {
      std::cout << "Actual output: " << std::endl;
      std::cout << actual_output << std::endl;
      std::cout << "Expected output: " << std::endl;
      std::cout << expected_output << std::endl;
    }
    ASSERT_TRUE(actual_output.isApprox(expected_output));
  }
}

TEST(DRSCertifierTest, GetBlockDiagOmega) {
  {
    // Case 1: N=10
    // read in expected data
    std::ifstream bdomega_source_file("./data/certification_test/case_1/block_diag_omega.csv");
    std::ifstream q_source_file("./data/certification_test/case_1/q.csv");
    auto q_mat = teaser::test::readFileToEigenFixedMatrix<double, 4, 1>(q_source_file);
    auto expected_output =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
            bdomega_source_file);
    double cbar2 = 1;
    double noise_bound = 4.594291399787397e-02;
    int N = 10;
    int Npm = 44;

    // construct the certifier
    teaser::DRSCertifier certifier(noise_bound, cbar2);
    Eigen::Quaternion<double> q(q_mat(0), q_mat(1), q_mat(2), q_mat(3));

    // perform the computation
    Eigen::MatrixXd actual_output;
    certifier.getBlockDiagOmega(Npm, q, &actual_output);
    if (!actual_output.isApprox(expected_output)) {
      std::cout << "Actual output: " << std::endl;
      std::cout << actual_output << std::endl;
      std::cout << "Expected output: " << std::endl;
      std::cout << expected_output << std::endl;
    }
    ASSERT_TRUE(actual_output.isApprox(expected_output));
  }
}

TEST(DRSCertifierTest, GetQCost) {
  {
    // Case 1: N=10
    // load parameters
    std::ifstream v1_source_file("./data/certification_test/case_1/v1.csv");
    std::ifstream v2_source_file("./data/certification_test/case_1/v2.csv");
    std::ifstream Q_cost_source_file("./data/certification_test/case_1/Q_cost.csv");
    auto v1 = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(v1_source_file);
    auto v2 = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(v2_source_file);
    auto expected_output =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
            Q_cost_source_file);
    double cbar2 = 1;
    double noise_bound = 4.594291399787397e-02;
    int N = 10;
    int Npm = 44;

    // construct the certifier
    teaser::DRSCertifier certifier(noise_bound, cbar2);

    // perform the computation
    Eigen::MatrixXd actual_output;
    certifier.getQCost(v1, v2, &actual_output);
    if (!actual_output.isApprox(expected_output)) {
      std::cout << "Actual output: " << std::endl;
      std::cout << actual_output << std::endl;
      std::cout << "Expected output: " << std::endl;
      std::cout << expected_output << std::endl;
    }
    ASSERT_TRUE(actual_output.isApprox(expected_output));
  }
}

TEST(DRSCertifierTest, GetLambdaGuess) { ASSERT_TRUE(false); }

TEST(DRSCertifierTest, GetLinearProjection) { ASSERT_TRUE(false); }
