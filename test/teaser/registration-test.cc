/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <random>

#include "gtest/gtest.h"

#include "teaser/registration.h"
#include "teaser/ply_io.h"
#include "test_utils.h"

TEST(RegistrationTest, LargeModel) {

  std::string model_file = "./data/registration_test/1000point_model.ply";
  std::string scene_file = "./data/registration_test/1000point_scene.ply";

  teaser::PLYReader reader;
  teaser::PointCloud src_cloud;
  auto status = reader.read(model_file, src_cloud);
  EXPECT_EQ(status, 0);
  auto eigen_src = teaser::test::teaserPointCloudToEigenMatrix<double>(src_cloud);

  teaser::PointCloud dst_cloud;
  status = reader.read(scene_file, dst_cloud);
  EXPECT_EQ(status, 0);
  auto eigen_dst = teaser::test::teaserPointCloudToEigenMatrix<double>(dst_cloud);

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  // Prepare the solver object
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = 0.0337;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
  params.rotation_cost_threshold = 0.005;
  params.max_clique_num_threads = 15;

  // Prepare the solver object
  teaser::RobustRegistrationSolver solver(params);

  // Solve
  solver.solve(eigen_src, eigen_dst);

  // Stop the timer
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time spent: " << duration.count() << std::endl;
}

TEST(RegistrationTest, LargeModelSingleThreaded) {

  std::string model_file = "./data/registration_test/1000point_model.ply";
  std::string scene_file = "./data/registration_test/1000point_scene.ply";

  teaser::PLYReader reader;
  teaser::PointCloud src_cloud;
  auto status = reader.read(model_file, src_cloud);
  EXPECT_EQ(status, 0);
  auto eigen_src = teaser::test::teaserPointCloudToEigenMatrix<double>(src_cloud);

  teaser::PointCloud dst_cloud;
  status = reader.read(scene_file, dst_cloud);
  EXPECT_EQ(status, 0);
  auto eigen_dst = teaser::test::teaserPointCloudToEigenMatrix<double>(dst_cloud);

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  // Prepare the solver object
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = 0.0337;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
  params.rotation_cost_threshold = 0.005;
  params.max_clique_num_threads = 1;

  // Prepare the solver object
  teaser::RobustRegistrationSolver solver(params);

  // Solve
  solver.solve(eigen_src, eigen_dst);

  // Stop the timer
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time spent: " << duration.count() << std::endl;
}

TEST(RegistrationTest, SolveForScale) {
  // Problem 1
  // Ref. scale: 0.955885
  {
    // read in input data
    std::ifstream objectFile("./data/registration_test/objectIn.csv");
    std::ifstream sceneFile("./data/registration_test/sceneIn.csv");
    double noise_bound = 0.0067364;

    // reference solution for scale
    double scale_est_ref = 0.955885;

    auto object_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(objectFile);
    auto scene_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(sceneFile);
    EXPECT_EQ(object_points.cols(), scene_points.cols());

    // Solve for scale
    auto start = std::chrono::high_resolution_clock::now();
    teaser::RobustRegistrationSolver solver;

    solver.setScaleEstimator(std::make_unique<teaser::TLSScaleSolver>(noise_bound, 1));

    Eigen::Matrix<int, 2, Eigen::Dynamic> object_map;
    Eigen::Matrix<int, 2, Eigen::Dynamic> scene_map;
    auto object_tims = solver.computeTIMs(object_points, &object_map);
    auto scene_tims = solver.computeTIMs(scene_points, &scene_map);
    solver.solveForScale(object_tims, scene_tims);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken to estimate scale: " << duration.count() << " microseconds."
              << std::endl;
    // Check solution
    auto scale_solution = solver.getSolution();
    EXPECT_NEAR(scale_solution.scale, scale_est_ref, 0.01);
  }
}

TEST(RegistrationTest, SolveForRotation) {
  // Robust solver to solve for rotation with FGR
  {
    // Set up parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.0067364;
    params.cbar2 = 1;
    params.estimate_scaling = true;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
    params.rotation_cost_threshold = 0.005;

    teaser::RobustRegistrationSolver solver(params);

    // Read in data
    std::ifstream source_file("./data/registration_test/rotation_only_src.csv");
    Eigen::Matrix<double, Eigen::Dynamic, 3> source_points =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, 3>(source_file);
    Eigen::Matrix<double, 3, Eigen::Dynamic> src = source_points.transpose();

    // Generate rotated points
    Eigen::Matrix3d expected_R;
    // clang-format off
    expected_R << 0.997379773225804, -0.019905935977315, -0.069551000516966,
                  0.013777311189888, 0.996068297974922, -0.087510750572249,
                  0.071019530105605, 0.086323226782879, 0.993732623426126;
    // clang-format on
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst = expected_R * src;

    // Solve & check for answer
    solver.solveForRotation(src, dst);
    auto rotation_solution = solver.getSolution();
    EXPECT_TRUE(teaser::test::getAngularError(expected_R, rotation_solution.rotation) < 1e-5);
  }
  // Robust solver to solve for rotation with QUATRO
  {
    // Set up parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.0067364;
    params.cbar2 = 1;
    params.estimate_scaling = true;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO;
    params.inlier_selection_mode ==
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;
    params.rotation_cost_threshold = 0.005;

    teaser::RobustRegistrationSolver solver(params);

    // Read in data
    std::ifstream source_file("./data/registration_test/rotation_only_src.csv");
    Eigen::Matrix<double, Eigen::Dynamic, 3> source_points =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, 3>(source_file);
    Eigen::Matrix<double, 3, Eigen::Dynamic> src = source_points.transpose();

    // Generate rotated points
    Eigen::Matrix3d expected_R;
    // clang-format off
    // Note that the Quatro only estimates relative yaw direction
    expected_R << 0.997379773225804, -0.072343541246221, 0.0,
                  0.072343541246221,  0.997379773225804, 0.0,
                                0.0,                0.0, 1.0;
    // clang-format on
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst = expected_R * src;

    // Solve & check for answer
    solver.solveForRotation(src, dst);
    auto rotation_solution = solver.getSolution();
    std::cout << teaser::test::getAngularError(expected_R, rotation_solution.rotation) << std::endl;
    EXPECT_TRUE(teaser::test::getAngularError(expected_R, rotation_solution.rotation) < 1e-5);
  }
  // Robust solver to solve for rotation with GNC-TLS
  {
    // Set up parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.0067364;
    params.cbar2 = 1;
    params.estimate_scaling = true;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;

    teaser::RobustRegistrationSolver solver(params);

    // Read in data
    std::ifstream source_file("./data/registration_test/rotation_only_src.csv");
    Eigen::Matrix<double, Eigen::Dynamic, 3> source_points =
        teaser::test::readFileToEigenMatrix<double, Eigen::Dynamic, 3>(source_file);
    Eigen::Matrix<double, 3, Eigen::Dynamic> src = source_points.transpose();

    // Generate rotated points
    Eigen::Matrix3d expected_R;
    // clang-format off
    expected_R << 0.997379773225804, -0.019905935977315, -0.069551000516966,
                  0.013777311189888, 0.996068297974922, -0.087510750572249,
                  0.071019530105605, 0.086323226782879, 0.993732623426126;
    // clang-format on
    Eigen::Matrix<double, 3, Eigen::Dynamic> dst = expected_R * src;

    // Solve & check answer
    solver.solveForRotation(src, dst);
    auto rotation_solution = solver.getSolution();
    EXPECT_TRUE(teaser::test::getAngularError(expected_R, rotation_solution.rotation) < 1e-5);
  }
}

TEST(RegistrationTest, SolveRegistrationProblemDecoupled) {
  // Registration problem 1
  // Ref. scale: 0.955885
  // TODO: Clean up this test
  {
    // Read in testing data from csv files
    std::ifstream objectFile("./data/registration_test/objectIn.csv");
    std::ifstream sceneFile("./data/registration_test/sceneIn.csv");
    auto object_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(objectFile);
    auto scene_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(sceneFile);
    EXPECT_TRUE(object_points.cols() != 0);
    EXPECT_EQ(object_points.cols(), scene_points.cols());

    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.0067364;
    params.cbar2 = 1;
    params.estimate_scaling = true;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
    params.rotation_cost_threshold = 0.005;

    // Prepare the solver object
    teaser::RobustRegistrationSolver solver(params);

    // Solve the registration problem
    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(object_points, scene_points);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken to solve a unknown-scale registration problem: " << duration.count()
              << " microseconds." << std::endl;

    // get the actual solution
    auto actual_solution = solver.getSolution();

    // reference solution for scale
    double expected_solution = 0.955885; // scale from Matlab example
    Eigen::Matrix3d expected_rotation_solution;
    expected_rotation_solution << 0.9974, -0.0199, -0.0696, 0.0138, 0.9961, -0.0875, 0.0710, 0.0863,
        0.9937;
    Eigen::Vector3d expected_translation_solution;
    expected_translation_solution << -0.1011, 0.0908, 0.1344;

    std::cout << "Rotation estimation error: "
              << teaser::test::getAngularError(expected_rotation_solution, actual_solution.rotation)
              << std::endl;
    std::cout << "Translation estimation error: "
              << (expected_translation_solution - actual_solution.translation).norm() << std::endl;
    EXPECT_NEAR(expected_solution, actual_solution.scale, 0.0001);
    EXPECT_LE(teaser::test::getAngularError(expected_rotation_solution, actual_solution.rotation),
              0.25);
    EXPECT_LE((expected_translation_solution - actual_solution.translation).norm(), 0.15);
  }
  // Registration problem 2
  // Fixed scale
  {
    // Read in testing data from csv files
    std::ifstream objectFile("./data/registration_test/objectIn.csv");
    std::ifstream sceneFile("./data/registration_test/sceneIn.csv");
    auto object_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(objectFile);
    auto scene_points = teaser::test::readFileToEigenMatrix<double, 3, Eigen::Dynamic>(sceneFile);
    EXPECT_TRUE(object_points.cols() != 0);
    EXPECT_EQ(object_points.cols(), scene_points.cols());

    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.0067364;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
    params.rotation_cost_threshold = 0.005;

    // Prepare the solver object
    teaser::RobustRegistrationSolver solver(params);

    auto start = std::chrono::high_resolution_clock::now();
    // solve the registration problem
    solver.solve(object_points, scene_points);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken to solve a known-scale registration problem: " << duration.count()
              << " microseconds." << std::endl;

    // reference solution for scale
    double expected_solution = 1; // scale from Matlab example
    Eigen::Matrix3d expected_rotation_solution;
    // clang-format off
    expected_rotation_solution << 0.9974, -0.0199, -0.0696,
                                  0.0138, 0.9961, -0.0875,
                                  0.0710, 0.0863, 0.9937;
    // clang-format on
    Eigen::Vector3d expected_translation_solution;
    expected_translation_solution << -0.1011, 0.0908, 0.1344;
    std::ifstream expected_inliers_file("./data/registration_test/fixed_scale_inliers.csv");
    auto expected_scale_inliers =
        teaser::test::readFileToEigenMatrix<bool, 1, Eigen::Dynamic>(expected_inliers_file);

    // get the actual solution & inliers
    auto actual_solution = solver.getSolution();
    auto actual_scale_inliers = solver.getScaleInliersMask();

    // compare expected with actual
    std::cout << "Rotation estimation error: "
              << teaser::test::getAngularError(expected_rotation_solution, actual_solution.rotation)
              << std::endl;
    std::cout << "Translation estimation error: "
              << (expected_translation_solution - actual_solution.translation).norm() << std::endl;
    /**
     * TODO: Update the expected inliers
    for (size_t i = 0; i < actual_scale_inliers.cols(); ++i) {
      EXPECT_TRUE(expected_scale_inliers(0, i) == actual_scale_inliers(0, i));
    }
    **/

    // rotation inlier mask and rotation inliers consistency
    const auto actual_rotation_inlier_mask = solver.getRotationInliersMask();
    const auto actual_rotation_inliers = solver.getRotationInliers();
    int count = 0;
    for (size_t i = 0; i < actual_rotation_inlier_mask.cols(); ++i) {
      if (actual_rotation_inlier_mask[i]) {
        count++;
      }
    }
    EXPECT_EQ(count, actual_rotation_inliers.size());

    EXPECT_NEAR(expected_solution, actual_solution.scale, 0.0001);
    EXPECT_LE(teaser::test::getAngularError(expected_rotation_solution, actual_solution.rotation),
              0.2);
    EXPECT_LE((expected_translation_solution - actual_solution.translation).norm(), 0.1);
  }
}

TEST(RegistrationTest, OutlierDetection) {

  // Random point cloud
  int N = 20;
  Eigen::Matrix<double, 3, Eigen::Dynamic> src =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
  Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
  src_h.resize(4, src.cols());
  src_h.topRows(3) = src;
  src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

  // An arbitrary transformation matrix
  Eigen::Matrix4d T;
  // clang-format off
  T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
       -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
       4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
       0,              0,                0,              1;
  // clang-format on

  // Apply transformation
  Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt = tgt_h.topRows(3);

  // Pick some points to be outliers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis1(1, 5);              // num outliers
  std::uniform_int_distribution<> dis2(0, tgt.cols() - 1); // pos of outliers
  std::uniform_int_distribution<> dis3(5, 10);             // random translation
  int N_OUTLIERS = dis1(gen);
  std::vector<bool> expected_outlier_mask(tgt.cols(), false);
  for (int i = 0; i < N_OUTLIERS; ++i) {
    int c_outlier_idx = dis2(gen);
    assert(c_outlier_idx < expected_outlier_mask.size());
    expected_outlier_mask[c_outlier_idx] = true;
    tgt.col(c_outlier_idx).array() += dis3(gen); // random translation
  }
  std::vector<int> expected_inliers;
  for (int i = 0; i < expected_outlier_mask.size(); ++i) {
    if (!expected_outlier_mask[i]) {
      expected_inliers.push_back(i);
    }
  }
  std::sort(expected_inliers.begin(), expected_inliers.end());

  // Prepare solver parameters
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = 0.001;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_cost_threshold = 0.005;

  // Solve with TEASER++
  teaser::RobustRegistrationSolver solver(params);
  solver.solve(src, tgt);

  auto solution = solver.getSolution();
  EXPECT_LE(teaser::test::getAngularError(T.topLeftCorner(3, 3), solution.rotation), 0.2);
  EXPECT_LE((T.topRightCorner(3, 1) - solution.translation).norm(), 0.1);

  auto final_inliers = solver.getInlierMaxClique();

  EXPECT_EQ(expected_inliers.size(), final_inliers.size());
  std::sort(expected_inliers.begin(), expected_inliers.end());
  std::sort(final_inliers.begin(), final_inliers.end());
  for (size_t i = 0; i < expected_inliers.size(); ++i) {
    EXPECT_EQ(expected_inliers[i], final_inliers[i]);
  }
}

TEST(RegistrationTest, NoMaxClique) {
  // Random point cloud
  int N = 20;
  Eigen::Matrix<double, 3, Eigen::Dynamic> src =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
  Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
  src_h.resize(4, src.cols());
  src_h.topRows(3) = src;
  src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

  // An arbitrary transformation matrix
  Eigen::Matrix4d T;
  // clang-format off
  T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
      -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
      4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
      0,              0,                0,              1;
  // clang-format on

  // Apply transformation
  Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt = tgt_h.topRows(3);

  // Pick some points to be outliers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis1(1, 5);              // num outliers
  std::uniform_int_distribution<> dis2(0, tgt.cols() - 1); // pos of outliers
  std::uniform_int_distribution<> dis3(5, 10);             // random translation
  int N_OUTLIERS = dis1(gen);
  std::vector<bool> expected_outlier_mask(tgt.cols(), false);
  for (int i = 0; i < N_OUTLIERS; ++i) {
    int c_outlier_idx = dis2(gen);
    assert(c_outlier_idx < expected_outlier_mask.size());
    expected_outlier_mask[c_outlier_idx] = true;
    tgt.col(c_outlier_idx).array() += dis3(gen); // random translation
  }
  std::vector<int> expected_inliers;
  for (int i = 0; i < expected_outlier_mask.size(); ++i) {
    if (!expected_outlier_mask[i]) {
      expected_inliers.push_back(i);
    }
  }
  std::sort(expected_inliers.begin(), expected_inliers.end());

  // Prepare solver parameters
  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = 0.01;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_cost_threshold = 0.005;
  params.use_max_clique = false;

  // Solve with TEASER++
  teaser::RobustRegistrationSolver solver(params);
  solver.solve(src, tgt);

  auto solution = solver.getSolution();
  EXPECT_LE(teaser::test::getAngularError(T.topLeftCorner(3, 3), solution.rotation), 0.2);
  EXPECT_LE((T.topRightCorner(3, 1) - solution.translation).norm(), 0.1);
}

TEST(RegistrationTest, CliqueFinderModes) {
  // Random point cloud
  int N = 20;
  Eigen::Matrix<double, 3, Eigen::Dynamic> src =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
  Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
  src_h.resize(4, src.cols());
  src_h.topRows(3) = src;
  src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

  // An arbitrary transformation matrix
  Eigen::Matrix4d T;
  // clang-format off
  T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
      -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
      4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
      0,              0,                0,              1;
  // clang-format on

  // Apply transformation
  Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt = tgt_h.topRows(3);

  // Pick some points to be outliers
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis1(1, 5);              // num outliers
  std::uniform_int_distribution<> dis2(0, tgt.cols() - 1); // pos of outliers
  std::uniform_int_distribution<> dis3(5, 10);             // random translation
  int N_OUTLIERS = dis1(gen);
  std::vector<bool> expected_outlier_mask(tgt.cols(), false);
  for (int i = 0; i < N_OUTLIERS; ++i) {
    int c_outlier_idx = dis2(gen);
    assert(c_outlier_idx < expected_outlier_mask.size());
    expected_outlier_mask[c_outlier_idx] = true;
    tgt.col(c_outlier_idx).array() += dis3(gen); // random translation
  }
  std::vector<int> expected_inliers;
  for (int i = 0; i < expected_outlier_mask.size(); ++i) {
    if (!expected_outlier_mask[i]) {
      expected_inliers.push_back(i);
    }
  }
  std::sort(expected_inliers.begin(), expected_inliers.end());

  {
    // PMC heuristic finder
    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.01;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;
    params.inlier_selection_mode =
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;

    // Solve with TEASER++
    teaser::RobustRegistrationSolver solver(params);
    solver.solve(src, tgt);

    auto solution = solver.getSolution();
    EXPECT_LE(teaser::test::getAngularError(T.topLeftCorner(3, 3), solution.rotation), 0.2);
    EXPECT_LE((T.topRightCorner(3, 1) - solution.translation).norm(), 0.1);
  }

  {
    // PMC heuristic finder
    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.01;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;
    params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;

    // Solve with TEASER++
    teaser::RobustRegistrationSolver solver(params);
    solver.solve(src, tgt);

    auto solution = solver.getSolution();
    EXPECT_LE(teaser::test::getAngularError(T.topLeftCorner(3, 3), solution.rotation), 0.2);
    EXPECT_LE((T.topRightCorner(3, 1) - solution.translation).norm(), 0.1);
  }

  {
    // K-core heuristic finder
    // Prepare solver parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.01;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;
    params.inlier_selection_mode =
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::KCORE_HEU;
    params.kcore_heuristic_threshold = 0.5;

    // Solve with TEASER++
    teaser::RobustRegistrationSolver solver(params);
    solver.solve(src, tgt);

    auto solution = solver.getSolution();
    EXPECT_LE(teaser::test::getAngularError(T.topLeftCorner(3, 3), solution.rotation), 0.2);
    EXPECT_LE((T.topRightCorner(3, 1) - solution.translation).norm(), 0.1);
  }
}
