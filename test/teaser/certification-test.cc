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
#include <algorithm>

#include <Eigen/Core>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "teaser/certification.h"
#include "test_utils.h"

/**
 * Acceptable numerical error threshold for all certification tests
 */
const double ACCEPTABLE_ERROR = 1e-7;

/**
 * Test fixture for loading data case by case
 */
class DRSCertifierTest : public ::testing::Test {
protected:
  /**
   * Inputs for the DRSCertifier
   *
   * Note that some may be intermediate outputs from other functions.
   */
  struct Inputs {
    double noise_bound;
    double cbar2;
    Eigen::Matrix<double, 3, Eigen::Dynamic> v1;
    Eigen::Matrix<double, 3, Eigen::Dynamic> v2;
    Eigen::Matrix3d R_est;
    Eigen::Quaternion<double> q_est;
    Eigen::Matrix<double, 1, Eigen::Dynamic> theta_est;
    Eigen::MatrixXd W;
    Eigen::MatrixXd M_affine;
    double mu;
  };

  /**
   * Expected outputs for methods under the DRSCertifier
   */
  struct ExpectedOutputs {
    Eigen::Matrix4d omega;
    Eigen::MatrixXd block_diag_omega;
    Eigen::MatrixXd Q_cost;
    Eigen::MatrixXd lambda_guess;
    Eigen::MatrixXd A_inv;
    Eigen::MatrixXd W_dual;
    double suboptimality_1st_iter;
    teaser::CertificationResult certification_result;
  };

  struct CaseData {
    Inputs inputs;
    ExpectedOutputs expected_outputs;
  };

  /**
   * Helper function to load parameters
   * @param file_path
   * @param cbar2
   * @param noise_bound
   */
  void loadScalarParameters(std::string file_path, double* cbar2, double* noise_bound) {
    // Open the file
    std::ifstream file;
    file.open(file_path);
    if (!file) {
      std::cerr << "Unable to open file: " << file_path << "." << std::endl;
      exit(1);
    }
    std::string line;
    std::string delimiter = ":";
    while (std::getline(file, line)) {

      size_t delim_idx = line.find(delimiter, 0);
      std::string param = line.substr(0, delim_idx);
      std::string value = line.substr(delim_idx + 2, line.length());

      if (param == "cbar2") {
        *cbar2 = std::stod(value);
      } else if (param == "noise_bound") {
        *noise_bound = std::stod(value);
      }
    }
  }

  /**
   * Helper function to compare two certification results
   * @param actual_result
   * @param expected_result
   */
  void compareCertificationResult(teaser::CertificationResult actual_result,
                                  teaser::CertificationResult expected_result) {

    EXPECT_EQ(actual_result.suboptimality_traj.size(), expected_result.suboptimality_traj.size())
        << "Mismatch in suboptimality trajectory sizes.";

    auto act_itr = actual_result.suboptimality_traj.begin();
    auto act_end_itr = actual_result.suboptimality_traj.end();
    auto exp_itr = expected_result.suboptimality_traj.begin();
    auto exp_end_itr = expected_result.suboptimality_traj.end();
    while (act_itr != act_end_itr && exp_itr != exp_end_itr &&
           std::abs(*act_itr - *exp_itr) < ACCEPTABLE_ERROR) {
      ++act_itr, ++exp_itr;
    }
    EXPECT_TRUE(act_itr == act_end_itr && exp_itr == exp_end_itr)
        << "Mismatch between actual and expected suboptimality trajectory values.";

    ASSERT_TRUE(std::abs(actual_result.best_suboptimality - expected_result.best_suboptimality) <
                ACCEPTABLE_ERROR)
        << "Incorrect best optimality gap.";
  }

  void SetUp() override {
    // load case parameters
    // read in all case folders
    std::string root_dir = "./data/certification_test/";
    auto cases = teaser::test::readSubdirs(root_dir);
    for (const auto& c : cases) {
      std::string case_dir = root_dir + c;

      CaseData data;

      // Inputs:
      // scalar parameters
      loadScalarParameters(case_dir + "/parameters.txt", &(data.inputs.cbar2),
                           &(data.inputs.noise_bound));

      // v1: 3-by-N matrix
      // These are the TIMs
      std::ifstream v1_source_file(case_dir + "/v1.csv");
      data.inputs.v1 =
          teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(v1_source_file);

      // v2: 3-by-N matrix
      // These are the TIMs
      std::ifstream v2_source_file(case_dir + "/v2.csv");
      data.inputs.v2 =
          teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(v2_source_file);

      // q_est: estimated quaternion
      std::ifstream q_source_file(case_dir + "/q_est.csv");
      auto q_mat = teaser::test::readFileToEigenFixedMatrix<double, 4, 1>(q_source_file);
      Eigen::Quaternion<double> q(q_mat(3), q_mat(0), q_mat(1), q_mat(2));
      data.inputs.q_est = q;

      // R_est: estimated quaternion
      std::ifstream R_est_source_file(case_dir + "/R_est.csv");
      data.inputs.R_est = teaser::test::readFileToEigenFixedMatrix<double, 3, 3>(R_est_source_file);

      // theta_est: binary outlier vector
      std::ifstream theta_source_file(case_dir + "/theta_est.csv");
      data.inputs.theta_est =
          teaser::test::readFileToEigenMatrix<double, 1, Eigen::Dynamic>(theta_source_file);

      // W: inputs for optimal dual projection
      std::ifstream W_source_file(case_dir + "/W_1st_iter.csv");
      data.inputs.W = teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
          W_source_file);

      // M_affine: for calculating suboptimality
      std::ifstream M_affine_source_file(case_dir + "/M_affine_1st_iter.csv");
      data.inputs.M_affine =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              M_affine_source_file);

      // mu: for calculating suboptimality
      std::ifstream mu_source_file(case_dir + "/mu.csv");
      data.inputs.mu = teaser::test::readFileToEigenFixedMatrix<double, 1, 1>(mu_source_file)(0);

      // Outputs:
      // omega: omega1 matrix
      std::ifstream omega_source_file(case_dir + "/omega.csv");
      data.expected_outputs.omega =
          teaser::test::readFileToEigenFixedMatrix<double, 4, 4>(omega_source_file);

      // block_diag_omega: block diagonal omega matrix
      std::ifstream block_diag_omega_source_file(case_dir + "/block_diag_omega.csv");
      data.expected_outputs.block_diag_omega =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              block_diag_omega_source_file);

      // Q_cost: Q_cost matrix
      std::ifstream q_cost_source_file(case_dir + "/Q_cost.csv");
      data.expected_outputs.Q_cost =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              q_cost_source_file);

      // lambda guess: initial guess
      std::ifstream lambda_guess_source_file(case_dir + "/lambda_bar_init.csv");
      data.expected_outputs.lambda_guess =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              lambda_guess_source_file);

      // A_inv: inverse map from getLinearProjection
      std::ifstream A_inv_source_file(case_dir + "/A_inv.csv");
      data.expected_outputs.A_inv =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              A_inv_source_file);

      // W_dual: output from optimal dual projection
      std::ifstream W_dual_source_file(case_dir + "/W_dual_1st_iter.csv");
      data.expected_outputs.W_dual =
          teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, Eigen::Dynamic>(
              W_dual_source_file);

      // suboptimality: calculated suboptimality after 1st iteration
      std::ifstream suboptimality_source_file(case_dir + "/suboptimality_1st_iter.csv");
      data.expected_outputs.suboptimality_1st_iter =
          teaser::test::readFileToEigenFixedMatrix<double, 1, 1>(suboptimality_source_file)(0);

      // certification_result: a struct holding certification results. Specifically:
      // suboptimality_trajy: suboptimality gaps throughout all the iterations
      // best_suboptimality: smallest suboptimality gap
      std::ifstream suboptimality_traj_source_file(case_dir + "/suboptimality_traj.csv");
      Eigen::RowVectorXd suboptimality_traj_mat =
          teaser::test::readFileToEigenMatrix<double, 1, Eigen::Dynamic>(
              suboptimality_traj_source_file);
      for (size_t i = 0; i < suboptimality_traj_mat.cols(); ++i) {
        data.expected_outputs.certification_result.suboptimality_traj.push_back(
            suboptimality_traj_mat(i));
      }
      data.expected_outputs.certification_result.best_suboptimality =
          suboptimality_traj_mat.minCoeff();

      case_params_[c] = data;
    }
  }

  // parameters per case
  std::unordered_map<std::string, CaseData> case_params_;
};

TEST_F(DRSCertifierTest, GetOmega1) {
  {
    // Case 1: N=10
    const auto& case_data = case_params_["case_1"];
    const auto& expected_output = case_data.expected_outputs.omega;

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.noise_bound, case_data.inputs.cbar2);

    // perform the computation
    Eigen::Matrix4d actual_output = certifier.getOmega1(case_data.inputs.q_est);
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  }
}

TEST_F(DRSCertifierTest, GetBlockDiagOmega) {
  {
    // Case 1: N=10
    const auto& case_data = case_params_["case_1"];
    const auto& expected_output = case_data.expected_outputs.block_diag_omega;

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.noise_bound, case_data.inputs.cbar2);

    // perform the computation
    Eigen::MatrixXd actual_output;
    int Npm = (case_data.inputs.v1.cols() + 1) * 4;
    certifier.getBlockDiagOmega(Npm, case_data.inputs.q_est, &actual_output);
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  }
}

TEST_F(DRSCertifierTest, GetQCost) {
  {
    // Case 1: N=10
    const auto& case_data = case_params_["case_1"];
    const auto& expected_output = case_data.expected_outputs.Q_cost;

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.noise_bound, case_data.inputs.cbar2);

    // perform the computation
    Eigen::MatrixXd actual_output;
    certifier.getQCost(case_data.inputs.v1, case_data.inputs.v2, &actual_output);
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  }
}

TEST_F(DRSCertifierTest, GetLambdaGuess) {
  {
    // Case 1: N=10
    const auto& case_data = case_params_["case_1"];

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.noise_bound, case_data.inputs.cbar2);

    Eigen::SparseMatrix<double> actual_output;
    certifier.getLambdaGuess(case_data.inputs.R_est, case_data.inputs.theta_est,
                             case_data.inputs.v1, case_data.inputs.v2, &actual_output);

    const auto& expected_output = case_data.expected_outputs.lambda_guess;
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  }
}

TEST_F(DRSCertifierTest, GetLinearProjection) {
  {
    // Case 1: N = 10
    const auto& case_data = case_params_["case_1"];

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.noise_bound, case_data.inputs.cbar2);

    Eigen::Matrix<double, 1, Eigen::Dynamic> theta_prepended(1,
                                                             case_data.inputs.theta_est.cols() + 1);
    theta_prepended << 1, case_data.inputs.theta_est;
    ASSERT_TRUE(theta_prepended.rows() == 1);
    ASSERT_TRUE(theta_prepended.cols() > 0);

    Eigen::SparseMatrix<double> actual_output;
    certifier.getLinearProjection(theta_prepended, &actual_output);

    const auto& expected_output = case_data.expected_outputs.A_inv;
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  }
}

TEST_F(DRSCertifierTest, GetOptimalDualProjection) {
  {
    // Case 1: N = 10
    const auto& case_data = case_params_["case_1"];

    // prepare parameters
    // theta prepended
    Eigen::Matrix<double, 1, Eigen::Dynamic> theta_prepended(1,
                                                             case_data.inputs.theta_est.cols() + 1);
    theta_prepended << 1, case_data.inputs.theta_est;
    ASSERT_TRUE(theta_prepended.rows() == 1);
    ASSERT_TRUE(theta_prepended.cols() > 0);

    // A_inv
    Eigen::SparseMatrix<double> A_inv_sparse = case_data.expected_outputs.A_inv.sparseView();

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.noise_bound, case_data.inputs.cbar2);

    Eigen::MatrixXd actual_output;
    certifier.getOptimalDualProjection(case_data.inputs.W, theta_prepended, A_inv_sparse,
                                       &actual_output);

    const auto& expected_output = case_data.expected_outputs.W_dual;
    for (size_t col = 0; col < expected_output.cols(); ++col) {
      for (size_t row = 0; row < expected_output.rows(); ++row) {
        if (std::abs(actual_output(row, col) - expected_output(row, col)) > ACCEPTABLE_ERROR) {
          std::cout << "Row: " << row << " Col: " << col << " Value: " << actual_output(row, col)
                    << " Expected: " << expected_output(row, col) << std::endl;
        }
      }
    }
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  }
}

TEST_F(DRSCertifierTest, ComputeSubOptimalityGap) {
  {
    // Case 1: N = 10
    const auto& case_data = case_params_["case_1"];

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.noise_bound, case_data.inputs.cbar2);

    double actual_output = certifier.computeSubOptimalityGap(
        case_data.inputs.M_affine, case_data.inputs.mu, case_data.inputs.v1.cols());
    const auto& expected_output = case_data.expected_outputs.suboptimality_1st_iter;

    ASSERT_TRUE(std::abs(actual_output - expected_output) < ACCEPTABLE_ERROR);
  }
}

TEST_F(DRSCertifierTest, Certify) {
  {
    // Case 1: N = 10
    const auto& case_data = case_params_["case_1"];

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.noise_bound, case_data.inputs.cbar2);

    auto actual_output = certifier.certify(case_data.inputs.R_est, case_data.inputs.v1,
                                           case_data.inputs.v2, case_data.inputs.theta_est);

    compareCertificationResult(actual_output, case_data.expected_outputs.certification_result);
  }
}
