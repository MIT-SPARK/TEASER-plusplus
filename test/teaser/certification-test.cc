/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <array>
#include <iostream>
#include <fstream>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <chrono>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

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
    teaser::DRSCertifier::Params params;
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
  void loadScalarParameters(std::string file_path, teaser::DRSCertifier::Params* params) {
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
        params->cbar2 = std::stod(value);
      } else if (param == "noise_bound") {
        params->noise_bound = std::stod(value);
      } else if (param == "max_iterations") {
        params->max_iterations = std::stod(value);
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

  /**
   * Helper function to set up small certification test instances
   */
  void setupSmallInstances() {
    std::string root_dir = "./data/certification_small_instances/";
    auto cases = teaser::test::readSubdirs(root_dir);
    for (const auto& c : cases) {
      std::string case_dir = root_dir + c;

      CaseData data;

      // Inputs:
      // scalar parameters
      loadScalarParameters(case_dir + "/parameters.txt", &(data.inputs.params));

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
      data.expected_outputs.certification_result.is_optimal = true;

      small_instances_params_[c] = data;
    }
  }

  /**
   * Helper function to load parameters for large problem instances
   *
   * Large instances are only used for testing the main certification function
   */
  void setupLargeInstances() {
    std::string root_dir = "./data/certification_large_instances/";
    auto cases = teaser::test::readSubdirs(root_dir);
    for (const auto& c : cases) {
      std::string case_dir = root_dir + c;

      CaseData data;

      // Inputs:
      // scalar parameters
      loadScalarParameters(case_dir + "/parameters.txt", &(data.inputs.params));

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
      data.expected_outputs.certification_result.is_optimal = true;

      large_instances_params_[c] = data;
    }
  }

  void SetUp() override {
    // load case parameters for small instances
    setupSmallInstances();

    // load case parameters for large instances
    setupLargeInstances();
  }

  /**
   * Helper function to run a provided function over a dictionary of case data
   * @tparam Functor
   * @param functor
   */
  template <typename Functor>
  void testThroughCases(const std::string& test_case_name, Functor functor,
                        const std::map<std::string, CaseData>& case_params) {
    // get all case names
    std::string ptr_str = "Timing info for test case: " + test_case_name;
    std::string div_str(ptr_str.size(), '=');
    std::cout << ptr_str << std::endl;
    std::cout << div_str << std::endl;
    for (auto const& kv : case_params) {
      const auto& case_name = kv.first;
      std::chrono::steady_clock clock;
      auto t1 = clock.now();
      functor(kv.second);
      auto t2 = clock.now();
      std::chrono::duration<double, std::milli> diff = t2 - t1;
      std::cout << "\nN=" << kv.second.inputs.v1.cols() << " | Test took "
                << static_cast<double>(diff.count()) / 1000.0 << "seconds." << std::endl;
    }
    std::cout << div_str << std::endl;
  }

  // parameters per case
  std::map<std::string, CaseData> small_instances_params_;
  std::map<std::string, CaseData> large_instances_params_;
};

TEST_F(DRSCertifierTest, GetOmega1) {
  auto test_run = [](CaseData case_data) {
    const auto& expected_output = case_data.expected_outputs.omega;

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    // perform the computation
    auto actual_output = certifier.getOmega1(case_data.inputs.q_est);
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  };

  testThroughCases(test_info_->name(), test_run, small_instances_params_);
}

TEST_F(DRSCertifierTest, GetBlockDiagOmega) {
  auto test_run = [](CaseData case_data) {
    // Case 1: N=10
    const auto& expected_output = case_data.expected_outputs.block_diag_omega;

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    // perform the computation
    Eigen::MatrixXd actual_output;
    int Npm = (case_data.inputs.v1.cols() + 1) * 4;
    certifier.getBlockDiagOmega(Npm, case_data.inputs.q_est, &actual_output);
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  };

  testThroughCases(test_info_->name(), test_run, small_instances_params_);
}

TEST_F(DRSCertifierTest, GetQCost) {
  auto test_run = [](CaseData case_data) {
    const auto& expected_output = case_data.expected_outputs.Q_cost;

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    // perform the computation
    Eigen::MatrixXd actual_output;
    certifier.getQCost(case_data.inputs.v1, case_data.inputs.v2, &actual_output);
    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  };

  testThroughCases(test_info_->name(), test_run, small_instances_params_);
}

TEST_F(DRSCertifierTest, GetLambdaGuess) {
  auto test_run = [](CaseData case_data) {
    const auto& expected_output = case_data.expected_outputs.lambda_guess;

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    teaser::SparseMatrix actual_output;
    certifier.getLambdaGuess(case_data.inputs.R_est, case_data.inputs.theta_est,
                             case_data.inputs.v1, case_data.inputs.v2, &actual_output);

    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  };

  testThroughCases(test_info_->name(), test_run, small_instances_params_);
}

TEST_F(DRSCertifierTest, GetLinearProjection) {
  auto test_run = [](CaseData case_data) {
    const auto& expected_output = case_data.expected_outputs.A_inv;

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    Eigen::Matrix<double, 1, Eigen::Dynamic> theta_prepended(1,
                                                             case_data.inputs.theta_est.cols() + 1);
    theta_prepended << 1, case_data.inputs.theta_est;
    ASSERT_TRUE(theta_prepended.rows() == 1);
    ASSERT_TRUE(theta_prepended.cols() > 0);

    teaser::SparseMatrix actual_output;
    certifier.getLinearProjection(theta_prepended, &actual_output);

    ASSERT_TRUE(actual_output.isApprox(expected_output))
        << "Actual output: " << actual_output << "Expected output: " << expected_output;
  };

  testThroughCases(test_info_->name(), test_run, small_instances_params_);
}

TEST_F(DRSCertifierTest, GetOptimalDualProjection) {
  auto test_run = [](CaseData case_data) {
    // prepare parameters
    // theta prepended
    Eigen::Matrix<double, 1, Eigen::Dynamic> theta_prepended(1,
                                                             case_data.inputs.theta_est.cols() + 1);
    theta_prepended << 1, case_data.inputs.theta_est;
    ASSERT_TRUE(theta_prepended.rows() == 1);
    ASSERT_TRUE(theta_prepended.cols() > 0);

    // A_inv
    teaser::SparseMatrix A_inv_sparse = case_data.expected_outputs.A_inv.sparseView();

    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

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
  };

  testThroughCases(test_info_->name(), test_run, small_instances_params_);
}

TEST_F(DRSCertifierTest, ComputeSubOptimalityGap) {
  auto test_run = [](CaseData case_data) {
    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    double actual_output = certifier.computeSubOptimalityGap(
        case_data.inputs.M_affine, case_data.inputs.mu, case_data.inputs.v1.cols());
    const auto& expected_output = case_data.expected_outputs.suboptimality_1st_iter;

    ASSERT_TRUE(std::abs(actual_output - expected_output) < ACCEPTABLE_ERROR);
  };

  testThroughCases(test_info_->name(), test_run, small_instances_params_);
}

TEST_F(DRSCertifierTest, Certify) {
  auto test_run = [&](CaseData case_data) {
    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    auto actual_output = certifier.certify(case_data.inputs.R_est, case_data.inputs.v1,
                                           case_data.inputs.v2, case_data.inputs.theta_est);

    compareCertificationResult(actual_output, case_data.expected_outputs.certification_result);
  };

  testThroughCases(test_info_->name(), test_run, small_instances_params_);
}

TEST_F(DRSCertifierTest, LargeInstance) {
  auto test_run = [&](CaseData case_data) {
    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    auto actual_output = certifier.certify(case_data.inputs.R_est, case_data.inputs.v1,
                                           case_data.inputs.v2, case_data.inputs.theta_est);

    compareCertificationResult(actual_output, case_data.expected_outputs.certification_result);
  };

  testThroughCases(test_info_->name(), test_run, large_instances_params_);
}

TEST_F(DRSCertifierTest, Random100Points) {
  // generate 3 random large problem instances
  std::map<std::string, CaseData> random_instances_params;
  int num_tests = 5;
  int N = 100;

  for (size_t i = 0; i < num_tests; ++i) {
    std::string case_name = "Random100-" + std::to_string(i) + "-" + std::to_string(N);

    CaseData data;

    // Inputs:
    // scalar parameters
    data.inputs.params.cbar2 = 1;
    data.inputs.params.noise_bound = 0.01;

    // generate random vectors and transformations
    data.inputs.v1 = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
    data.inputs.q_est = Eigen::Quaternion<double>::UnitRandom();
    data.inputs.R_est = data.inputs.q_est.toRotationMatrix();
    data.inputs.theta_est = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1, N);

    // calculate vectors after transformation
    // noise bounded by 0.01
    Eigen::Matrix<double, 3, Eigen::Dynamic> noise =
        (Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N).array() + 1) / 200.0;
    data.inputs.v2 = data.inputs.R_est * data.inputs.v1;

    // outliers
    double outlier_ratio = 0.1;
    int outlier_start_idx = (int)(N * (1 - outlier_ratio));
    for (size_t i = outlier_start_idx; i < N; ++i) {
      data.inputs.v2.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1) * 5 +
                              Eigen::Matrix<double, 3, Eigen::Dynamic>::Ones(3, 1) * 5;
      data.inputs.theta_est(0, i) = -1;
    }

    // expected outputs
    data.expected_outputs.certification_result.is_optimal = true;
    data.expected_outputs.certification_result.best_suboptimality = 1e-5; // a very small value

    random_instances_params[case_name] = data;
  }

  auto test_run = [](CaseData case_data) {
    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    auto actual_output = certifier.certify(case_data.inputs.R_est, case_data.inputs.v1,
                                           case_data.inputs.v2, case_data.inputs.theta_est);
    const auto& expected_output = case_data.expected_outputs.certification_result;

    ASSERT_TRUE(expected_output.is_optimal == actual_output.is_optimal);
    ASSERT_TRUE(expected_output.best_suboptimality >= actual_output.best_suboptimality);
  };

  testThroughCases(test_info_->name(), test_run, random_instances_params);
}

TEST_F(DRSCertifierTest, RandomLargeInstsances) {
  // generate 3 random large problem instances
  std::map<std::string, CaseData> random_instances_params;
  std::array<double, 3> problem_sizes = {200, 300, 400};

  for (size_t i = 0; i < problem_sizes.size(); ++i) {

    int N = problem_sizes.at(i);
    std::string case_name = "Random-" + std::to_string(N);

    CaseData data;

    // Inputs:
    // scalar parameters
    data.inputs.params.cbar2 = 1;
    data.inputs.params.noise_bound = 0.01;

    // generate random vectors and transformations
    data.inputs.v1 = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
    data.inputs.q_est = Eigen::Quaternion<double>::UnitRandom();
    data.inputs.R_est = data.inputs.q_est.toRotationMatrix();
    data.inputs.theta_est = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(1, N);

    // calculate vectors after transformation
    // noise bounded by 0.01
    Eigen::Matrix<double, 3, Eigen::Dynamic> noise =
        (Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N).array() + 1) / 200.0;
    data.inputs.v2 = data.inputs.R_est * data.inputs.v1;

    // outliers
    double outlier_ratio = 0.1;
    int outlier_start_idx = (int)(N * (1 - outlier_ratio));
    for (size_t i = outlier_start_idx; i < N; ++i) {
      data.inputs.v2.col(i) = Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, 1) * 5 +
                              Eigen::Matrix<double, 3, Eigen::Dynamic>::Ones(3, 1) * 5;
      data.inputs.theta_est(0, i) = -1;
    }

    // expected outputs
    data.expected_outputs.certification_result.is_optimal = true;
    data.expected_outputs.certification_result.best_suboptimality = 1e-5; // a very small value

    random_instances_params[case_name] = data;
  }

  auto test_run = [](CaseData case_data) {
    // construct the certifier
    teaser::DRSCertifier certifier(case_data.inputs.params);

    auto actual_output = certifier.certify(case_data.inputs.R_est, case_data.inputs.v1,
                                           case_data.inputs.v2, case_data.inputs.theta_est);
    const auto& expected_output = case_data.expected_outputs.certification_result;

    ASSERT_TRUE(expected_output.is_optimal == actual_output.is_optimal);
    ASSERT_TRUE(expected_output.best_suboptimality >= actual_output.best_suboptimality);
  };

  testThroughCases(test_info_->name(), test_run, random_instances_params);
}
