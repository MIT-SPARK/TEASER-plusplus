/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "teaser/registration.h"
#include "teaser/ply_io.h"
#include "test_utils.h"

/**
 * This file contains a small framework for running benchmark with specifications.
 *
 * By providing a parameters.txt file with relevant parameters, and relevant data files, this
 * framework will run, time and print out the performance of the solver under test.
 */
class RegistrationBenchmark : public ::testing::Test {
protected:
  /**
   * Enum representing the types of parameters we have
   */
  enum class BenchmarkParamType { NUM_POINTS, NOISE_SIGMA, OUTLIER_RATIO, NOISE_BOUND };

  /**
   * Struct to represent all necessary data / reference values for a benchmark
   */
  struct BenchmarkData {
    Eigen::Matrix<double, 3, Eigen::Dynamic> src;
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst;
    double s_est;
    double s_ref;
    Eigen::Matrix3d R_est;
    Eigen::Matrix3d R_ref;
    Eigen::Vector3d t_est;
    Eigen::Vector3d t_ref;
    double noise_sigma;
    double noise_bound;
    double outlier_ratio;
    int num_points;
  };

  /**
   * Struct to store acceptable errors
   */
  struct ErrorConditions {
    // acceptable errors from ground truths
    double s_ground_truth_error;
    double R_ground_truth_error;
    double t_ground_truth_error;

    // acceptable errors from MATLAB TEASER implementation
    double s_TEASER_error;
    double R_TEASER_error;
    double t_TEASER_error;
  };

  /**
   * Map from string to BenchmarkParam
   */
  std::map<std::string, BenchmarkParamType> mapStringToBenchmarkParamType = {
      {"Number of Points", BenchmarkParamType::NUM_POINTS},
      {"Noise Sigma", BenchmarkParamType::NOISE_SIGMA},
      {"Outlier Ratio", BenchmarkParamType::OUTLIER_RATIO},
      {"Noise Bound", BenchmarkParamType::NOISE_BOUND},
  };

  /**
   * Helper function to load a parameters.txt file
   * @param num_points
   * @param noise_sigma
   * @param outlier_ratio
   * @param noise_bound
   */
  void loadParameters(std::string file_path, int* num_points, double* noise_sigma,
                      double* outlier_ratio, double* noise_bound) {
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

      BenchmarkParamType param_type = mapStringToBenchmarkParamType[param];
      switch (param_type) {
      case BenchmarkParamType::NUM_POINTS:
        *num_points = std::stoi(value);
        break;
      case BenchmarkParamType::NOISE_SIGMA:
        *noise_sigma = std::stod(value);
        break;
      case BenchmarkParamType::OUTLIER_RATIO:
        *outlier_ratio = std::stod(value);
        break;
      case BenchmarkParamType::NOISE_BOUND:
        *noise_bound = std::stod(value);
        break;
      }
    }
  }

  /**
   * Load the benchmark data for a specific benchmark
   * @param name the name of the benchmark, should correspond to one of the folder in the benchmark
   * folder
   */
  BenchmarkData prepareBenchmarkData(std::string name) {
    // Generate file names & file streams
    std::string folder = "./data/" + name + "/";
    std::string dst_file = folder + "dst.ply";
    std::string src_file = folder + "src.ply";
    std::string parameters_file = folder + "parameters.txt";
    std::ifstream R_ref_file(folder + "R_ref.csv");
    std::ifstream R_est_file(folder + "R_est.csv");
    std::ifstream s_ref_file(folder + "s_ref.csv");
    std::ifstream s_est_file(folder + "s_est.csv");
    std::ifstream t_ref_file(folder + "t_ref.csv");
    std::ifstream t_est_file(folder + "t_est.csv");

    // Struct to store all data & parameters
    BenchmarkData benchmark_data;

    // Load src model
    teaser::PLYReader reader;
    teaser::PointCloud src_cloud;
    auto status = reader.read(src_file, src_cloud);
    EXPECT_EQ(status, 0);
    benchmark_data.src = teaser::test::teaserPointCloudToEigenMatrix<double>(src_cloud);

    // Load dst model
    teaser::PointCloud dst_cloud;
    status = reader.read(dst_file, dst_cloud);
    EXPECT_EQ(status, 0);
    benchmark_data.dst = teaser::test::teaserPointCloudToEigenMatrix<double>(dst_cloud);

    // Load parameters
    double noise_bound, outlier_ratio, noise_sigma;
    loadParameters(parameters_file, &(benchmark_data.num_points), &(benchmark_data.noise_sigma),
                   &(benchmark_data.outlier_ratio), &(benchmark_data.noise_bound));

    // Load reference values
    benchmark_data.R_est = teaser::test::readFileToEigenFixedMatrix<double, 3, 3>(R_est_file);
    benchmark_data.R_ref = teaser::test::readFileToEigenFixedMatrix<double, 3, 3>(R_ref_file);
    benchmark_data.t_est = teaser::test::readFileToEigenFixedMatrix<double, 3, 1>(t_est_file);
    benchmark_data.t_ref = teaser::test::readFileToEigenFixedMatrix<double, 3, 1>(t_ref_file);
    s_est_file >> benchmark_data.s_est;
    s_ref_file >> benchmark_data.s_ref;

    return benchmark_data;
  }

  /**
   * Helper function for running a specific benchmark
   * @param folder
   */
  void benchmarkRunner(BenchmarkData data, ErrorConditions conditions,
                       std::string rotation_method = "GNC-TLS", size_t num_runs = 100) {

    // Variables for storing average errors
    double s_err_ref_avg = 0, t_err_ref_avg = 0, R_err_ref_avg = 0, s_err_est_avg = 0,
           t_err_est_avg = 0, R_err_est_avg = 0;
    double duration_avg = 0;

    for (size_t i = 0; i < num_runs; ++i) {
      // Start the timer
      auto start = std::chrono::high_resolution_clock::now();

      teaser::RobustRegistrationSolver::Params params;
      params.noise_bound = data.noise_bound;
      params.cbar2 = 1;
      params.estimate_scaling = true;
      params.rotation_max_iterations = 100;
      params.rotation_gnc_factor = 1.4;
      if (rotation_method == "GNC-TLS") {
        params.rotation_estimation_algorithm =
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
        params.rotation_cost_threshold = 1e-12;
      } else if (rotation_method == "FGR") {
        params.rotation_estimation_algorithm =
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
        params.rotation_cost_threshold = 0.005;
      } else {
        std::cout << "Unsupported rotation estimation method." << std::endl;
        break;
      }

      // Prepare the solver object
      teaser::RobustRegistrationSolver solver(params);

      // Solve
      solver.solve(data.src, data.dst);

      // Stop the timer
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      duration_avg += duration.count();

      // Get the solution
      auto actual_solution = solver.getSolution();

      // Errors wrt ground truths
      double s_err_ref = std::abs(actual_solution.scale - data.s_ref);
      double t_err_ref = (actual_solution.translation - data.t_ref).norm();
      double R_err_ref = teaser::test::getAngularError(data.R_ref, actual_solution.rotation);
      EXPECT_LE(s_err_ref, conditions.s_ground_truth_error);
      EXPECT_LE(t_err_ref, conditions.t_ground_truth_error);
      EXPECT_LE(R_err_ref, conditions.R_ground_truth_error);
      s_err_ref_avg += s_err_ref;
      t_err_ref_avg += t_err_ref;
      R_err_ref_avg += R_err_ref;

      // Errors wrt MATLAB implementation (TEASER w/ SDP rotation estimation) output
      double s_err_est = std::abs(actual_solution.scale - data.s_est);
      double t_err_est = (actual_solution.translation - data.t_est).norm();
      double R_err_est = teaser::test::getAngularError(data.R_est, actual_solution.rotation);
      EXPECT_LE(s_err_est, conditions.s_TEASER_error);
      EXPECT_LE(t_err_est, conditions.t_TEASER_error);
      EXPECT_LE(R_err_est, conditions.R_TEASER_error);
      s_err_est_avg += s_err_est;
      t_err_est_avg += t_err_est;
      R_err_est_avg += R_err_est;
    }
    double div_factor = 1.0 / static_cast<double>(num_runs);
    s_err_ref_avg *= div_factor;
    R_err_ref_avg *= div_factor;
    t_err_ref_avg *= div_factor;
    s_err_est_avg *= div_factor;
    R_err_est_avg *= div_factor;
    t_err_est_avg *= div_factor;
    duration_avg *= div_factor;

    // Print report
    std::cout << "==============================================" << std::endl;
    std::cout << "               Benchmark Report               " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "             Benchmark Parameters             " << std::endl;
    std::cout << "  num of points: " << data.num_points << std::endl;
    std::cout << "  outlier ratio: " << data.outlier_ratio << std::endl;
    std::cout << "    noise sigma: " << data.noise_sigma << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "            Error from Ground Truth           " << std::endl;
    std::cout << "       in scale: " << s_err_ref_avg << std::endl;
    std::cout << "    in rotation: " << R_err_ref_avg << std::endl;
    std::cout << " in translation: " << t_err_ref_avg << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << " Error from TEASER w/ SDP Rotation Estimation " << std::endl;
    std::cout << "       in scale: " << s_err_est_avg << std::endl;
    std::cout << "    in rotation: " << R_err_est_avg << std::endl;
    std::cout << " in translation: " << t_err_est_avg << std::endl;
    std::cout << "==============================================" << std::endl;

    std::cout << "Time taken to run benchmark: " << duration_avg << " microseconds." << std::endl;
  }

  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(RegistrationBenchmark, Benchmark1) {
  auto data = prepareBenchmarkData("benchmark_1");
  // Prepare acceptable errors
  ErrorConditions conditions;
  conditions.s_ground_truth_error = 1e-5;
  conditions.R_ground_truth_error = 1e-5;
  conditions.t_ground_truth_error = 1e-5;
  conditions.s_TEASER_error = 1e-5;
  conditions.R_TEASER_error = 1e-5;
  conditions.t_TEASER_error = 1e-5;

  // Run benchmark
  benchmarkRunner(data, conditions, "GNC-TLS");
  benchmarkRunner(data, conditions, "FGR");
}

TEST_F(RegistrationBenchmark, Benchmark2) {
  auto data = prepareBenchmarkData("benchmark_2");
  std::cout << data.src << std::endl;
  // Prepare acceptable errors
  ErrorConditions conditions;
  conditions.s_ground_truth_error = 1e-5;
  conditions.R_ground_truth_error = 1e-5;
  conditions.t_ground_truth_error = 1e-5;
  conditions.s_TEASER_error = 1e-5;
  conditions.R_TEASER_error = 1e-5;
  conditions.t_TEASER_error = 1e-5;

  // Run benchmark
  benchmarkRunner(data, conditions, "GNC-TLS");
  benchmarkRunner(data, conditions, "FGR");
}

TEST_F(RegistrationBenchmark, Benchmark3) {
  auto data = prepareBenchmarkData("benchmark_3");
  // Prepare acceptable errors
  ErrorConditions conditions;
  conditions.s_ground_truth_error = 1e-5;
  conditions.R_ground_truth_error = 1e-5;
  conditions.t_ground_truth_error = 1e-5;
  conditions.s_TEASER_error = 1e-5;
  conditions.R_TEASER_error = 1e-5;
  conditions.t_TEASER_error = 1e-5;

  // Run benchmark
  benchmarkRunner(data, conditions, "GNC-TLS");
  benchmarkRunner(data, conditions, "FGR");
}

TEST_F(RegistrationBenchmark, Benchmark4) {
  auto data = prepareBenchmarkData("benchmark_4");
  // Prepare acceptable errors
  ErrorConditions conditions;
  conditions.s_ground_truth_error = 1e-5;
  conditions.R_ground_truth_error = 1e-5;
  conditions.t_ground_truth_error = 1e-5;
  conditions.s_TEASER_error = 1e-5;
  conditions.R_TEASER_error = 1e-5;
  conditions.t_TEASER_error = 1e-5;

  // Run benchmark
  benchmarkRunner(data, conditions, "GNC-TLS");
  benchmarkRunner(data, conditions, "FGR");
}

TEST_F(RegistrationBenchmark, Benchmark5) {
  auto data = prepareBenchmarkData("benchmark_5");
  // Prepare acceptable errors
  ErrorConditions conditions;
  conditions.s_ground_truth_error = 1e-5;
  conditions.R_ground_truth_error = 1e-5;
  conditions.t_ground_truth_error = 1e-5;
  conditions.s_TEASER_error = 1e-5;
  conditions.R_TEASER_error = 1e-5;
  conditions.t_TEASER_error = 1e-5;

  // Run benchmark
  benchmarkRunner(data, conditions, "GNC-TLS");
  benchmarkRunner(data, conditions, "FGR");
}

/**
 * This test assumes non-zero noise and non-zero outlier ratio.
 */
TEST_F(RegistrationBenchmark, Benchmark6) {
  auto data = prepareBenchmarkData("benchmark_6");
  // Prepare acceptable errors
  ErrorConditions conditions;
  conditions.s_ground_truth_error = 1e-2;
  conditions.R_ground_truth_error = 1e-2;
  conditions.t_ground_truth_error = 2e-2;
  conditions.s_TEASER_error = 1e-5;
  conditions.R_TEASER_error = 1e-3;
  conditions.t_TEASER_error = 1e-3;

  // Run benchmark
  benchmarkRunner(data, conditions, "GNC-TLS");
  benchmarkRunner(data, conditions, "FGR");
}
