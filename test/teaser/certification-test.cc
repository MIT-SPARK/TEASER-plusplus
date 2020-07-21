/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <iostream>
#include <fstream>
#include <random>
#include <unordered_map>

#include <Eigen/Core>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "teaser/certification.h"
#include "test_utils.h"

/**
 * Test fixture for loading data case by case
 */
class DRSCertifierTest : public ::testing::Test {
protected:
  struct Inputs {
    Eigen::Matrix<double, 3, Eigen::Dynamic> v1;
    Eigen::Matrix<double, 3, Eigen::Dynamic> v2;
    Eigen::Matrix3d R_est;
    Eigen::Quaternion<double> q_est;
    Eigen::Matrix<double, 1, Eigen::Dynamic> theta_est;
  };

  struct ExpectedOutputs {
    Eigen::Matrix4d omega;
    Eigen::MatrixXd block_diag_omega;
    Eigen::MatrixXd Q_cost;
    Eigen::MatrixXd lambda_guess;
  };

  struct CaseData {
    Inputs inputs;
    ExpectedOutputs expected_outputs;
  };

  void SetUp() override {
    // prepare parameters
    noise_bound_ = 4.594291399787397e-02;
    cbar2_ = 1;

    // load case parameters
    // read in all case folders
    std::string root_dir = "./data/certification_test/";
    auto cases = teaser::test::readSubdirs(root_dir);
    for (const auto& c : cases) {
      std::string case_dir = root_dir + c;

      CaseData data;

      // inputs
      // v1: 3-by-N matrix
      std::ifstream v1_source_file("./data/certification_test/case_1/v1.csv");
      data.inputs.v1 =
          teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(v1_source_file);

      // v2: 3-by-N matrix
      std::ifstream v2_source_file("./data/certification_test/case_1/v2.csv");
      data.inputs.v2 =
          teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(v2_source_file);

      // q_est: estimated quaternion
      std::ifstream q_source_file("./data/certification_test/case_1/q_est.csv");
      auto q_mat = teaser::test::readFileToEigenFixedMatrix<double, 4, 1>(q_source_file);
      Eigen::Quaternion<double> q(q_mat(0), q_mat(1), q_mat(2), q_mat(3));
      data.inputs.q_est = q;

      // R_est: estimated quaternion
      std::ifstream R_est_source_file("./data/certification_test/case_1/R_est.csv");
      data.inputs.R_est = teaser::test::readFileToEigenFixedMatrix<double, 3, 3>(R_est_source_file);

      // theta_est: binary outlier vector
      std::ifstream theta_source_file("./data/certification_test/case_1/theta_est.csv");
      data.inputs.theta_est =
          teaser::test::readFileToEigenMatrix<double, 1, Eigen::Dynamic>(theta_source_file);
      std::cout << "Theta est: " << data.inputs.theta_est << std::endl;

      // omega: omega1 matrix
      std::ifstream omega_source_file("./data/certification_test/case_1/omega.csv");
      data.expected_outputs.omega =
          teaser::test::readFileToEigenFixedMatrix<double, 4, 4>(omega_source_file);

      // block_diag_omega: block diagonal omega matrix
      std::ifstream block_diag_omega_source_file(
          "./data/certification_test/case_1/block_diag_omega.csv");
      data.expected_outputs.block_diag_omega =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              block_diag_omega_source_file);

      // Q_cost: Q_cost matrix
      std::ifstream q_cost_source_file("./data/certification_test/case_1/Q_cost.csv");
      data.expected_outputs.Q_cost =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              q_cost_source_file);

      // lambda guess: initial guess
      std::ifstream lambda_guess_source_file("./data/certification_test/case_1/lambda_bar_init.csv");
      data.expected_outputs.lambda_guess =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              lambda_guess_source_file);

      case_params_[c] = data;
    }
  }

  // parameters per case
  std::unordered_map<std::string, CaseData> case_params_;
  double cbar2_;
  double noise_bound_;
};

TEST_F(DRSCertifierTest, GetOmega1) {
  {
    // Case 1: N=10
    // read in expected data
    std::ifstream omega_source_file("./data/certification_test/case_1/omega.csv");
    std::ifstream q_source_file("./data/certification_test/case_1/q_est.csv");
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

TEST_F(DRSCertifierTest, GetBlockDiagOmega) {
  {
    // Case 1: N=10
    // read in expected data
    std::ifstream bdomega_source_file("./data/certification_test/case_1/block_diag_omega.csv");
    std::ifstream q_source_file("./data/certification_test/case_1/q_est.csv");
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

TEST_F(DRSCertifierTest, GetQCost) {
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

TEST_F(DRSCertifierTest, GetLambdaGuess) {
  {
    // Case 1: N=10
    const auto& case_data = case_params_["case_1"];

    // construct the certifier
    teaser::DRSCertifier certifier(noise_bound_, cbar2_);

    Eigen::SparseMatrix<double> actual_output;
    certifier.getLambdaGuess(case_data.inputs.R_est, case_data.inputs.theta_est,
                             case_data.inputs.v1, case_data.inputs.v2, &actual_output);

    if (!actual_output.isApprox(case_data.expected_outputs.lambda_guess)) {
      std::cout << "Actual output: " << std::endl;
      std::cout << actual_output << std::endl;
      std::cout << "Expected output: " << std::endl;
      std::cout << case_data.expected_outputs.lambda_guess << std::endl;
    }
    ASSERT_TRUE(actual_output.isApprox(case_data.expected_outputs.lambda_guess));
  }
}

TEST_F(DRSCertifierTest, GetLinearProjection) { ASSERT_TRUE(false); }
