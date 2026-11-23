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
#include <limits>
#include <random>
#include <stdexcept>

#include "gtest/gtest.h"

#include "teaser/registration.h"
#include "teaser/ply_io.h"
#include "test_utils.h"

TEST(RegistrationTest, ExplicitNoiseBoundKeepsSingleSolveBehavior) {
  using Solver = teaser::RobustRegistrationSolver;
  Eigen::Matrix<double, 3, Eigen::Dynamic> source(3, 5);
  source << 0, 1, 0, 0, 3,
            0, 0, 2, 0, 1,
            0, 0, 0, 4, 2;
  const auto rotation = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix<double, 3, Eigen::Dynamic> target =
      (1.5 * rotation * source).colwise() + Eigen::Vector3d(1, 2, 3);
  Solver::Params params;
  params.noise_bound = 0.1;
  params.max_clique_num_threads = 1;
  const auto expected = Solver(params).solve(source, target);
  ASSERT_TRUE(expected.valid);
  Solver solver(params);
  const auto scalar = solver.solve(source, target, 0.1);
  const auto list = solver.solve(source, target, std::vector<double>{0.1});
  const auto sweep = solver.solve(source, target, std::vector<double>{0.1, 0.05});
  for (const auto& result : {scalar, list, sweep}) {
    EXPECT_TRUE(result.valid);
    EXPECT_DOUBLE_EQ(result.scale, expected.scale);
    EXPECT_TRUE(result.rotation.isApprox(expected.rotation));
    EXPECT_TRUE(result.translation.isApprox(expected.translation));
  }
  EXPECT_THROW(solver.solve(source, target, std::vector<double>{}), std::invalid_argument);
  EXPECT_THROW(solver.solve(source, target, std::vector<double>{0.1, -1}), std::invalid_argument);
  EXPECT_TRUE(solver.getSolution().translation.isApprox(expected.translation));
}

TEST(RegistrationTest, AutomaticNoiseBoundsUseTargetSpacing) {
  using Solver = teaser::RobustRegistrationSolver;
  Eigen::Matrix<double, 3, Eigen::Dynamic> source(3, 4);
  source << 0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
  const Eigen::Matrix<double, 3, Eigen::Dynamic> target =
      (2.0 * source).colwise() + Eigen::Vector3d(3, -2, 1);
  Solver::Params params;
  params.estimate_scaling = true;
  params.max_clique_num_threads = 1;
  const std::vector<double> bounds{2, 4, 8};
  Solver manual(params), automatic(params);
  const auto expected = manual.solve(source, target, bounds);
  const auto actual = automatic.solve(source, target, Solver::NoiseBoundMode::AUTO);
  ASSERT_TRUE(expected.valid);
  ASSERT_TRUE(actual.valid);
  EXPECT_DOUBLE_EQ(automatic.getParams().noise_bound, manual.getParams().noise_bound);
  EXPECT_EQ(automatic.getInlierMaxClique(), manual.getInlierMaxClique());
  EXPECT_EQ(actual.rotation, expected.rotation);
  EXPECT_EQ(actual.translation, expected.translation);

  teaser::PointCloud source_cloud, target_cloud;
  std::vector<std::pair<int, int>> matches;
  for (int i = 0; i < source.cols(); ++i) {
    source_cloud.push_back({float(source(0, i)), float(source(1, i)), float(source(2, i))});
    target_cloud.push_back({float(target(0, i)), float(target(1, i)), float(target(2, i))});
    matches.emplace_back(i, i);
  }
  const auto cloud_pose = automatic.solve(source_cloud, target_cloud, matches,
                                          Solver::NoiseBoundMode::AUTO);
  ASSERT_TRUE(cloud_pose.valid);
  EXPECT_DOUBLE_EQ(automatic.getParams().noise_bound, manual.getParams().noise_bound);
  EXPECT_EQ(cloud_pose.rotation, expected.rotation);
  EXPECT_EQ(cloud_pose.translation, expected.translation);

  const Eigen::Matrix<double, 3, Eigen::Dynamic> verification_target =
      target.colwise() + Eigen::Vector3d(3e6, 0, 0);
  const auto dense_pose = automatic.solve(source, target, Solver::NoiseBoundMode::AUTO,
                                          source, verification_target);
  ASSERT_TRUE(dense_pose.valid);
  EXPECT_DOUBLE_EQ(automatic.getParams().noise_bound, manual.getParams().noise_bound);

  // Only matched indices contribute when no verification clouds are passed.
  target_cloud.push_back({std::numeric_limits<float>::quiet_NaN(), 0, 0});
  EXPECT_TRUE(automatic.solve(source_cloud, target_cloud, matches, 0.5).valid);
  EXPECT_TRUE(automatic.solve(source_cloud, target_cloud, matches, std::vector<double>{0.5}).valid);
  EXPECT_TRUE(automatic.solve(source_cloud, target_cloud, matches, bounds).valid);
  const auto invalid_cloud = teaser::test::teaserPointCloudToEigenMatrix<double>(target_cloud);
  EXPECT_THROW(automatic.solve(source, target, bounds, source, invalid_cloud),
               std::invalid_argument);
}

TEST(RegistrationTest, MultipleNoiseBoundsCanScoreSeparateVerificationClouds) {
  using Solver = teaser::RobustRegistrationSolver;
  std::mt19937 rng(13);
  std::uniform_real_distribution<float> position(-5, 5);
  std::uniform_real_distribution<float> jitter(-0.12, 0.12);
  teaser::PointCloud source, target;
  for (int i = 0; i < 256; ++i) {
    const teaser::PointXYZ point{position(rng), position(rng), position(rng)};
    source.push_back(point);
    target.push_back({point.x + 1.25f, point.y - 2.5f, point.z + 0.75f});
  }
  std::vector<std::pair<int, int>> matches;
  for (int i = 0; i < 16; ++i) {
    matches.emplace_back(i, i);
  }
  for (int i = 0; i < 24; ++i) {
    const auto& point = source[16 + i];
    target.push_back({point.x + 25 + jitter(rng), point.y + jitter(rng), point.z + jitter(rng)});
    matches.emplace_back(16 + i, 256 + i);
  }
  Solver::Params params;
  params.estimate_scaling = false;
  params.max_clique_num_threads = 1;
  Solver tight_solver(params), solver(params);
  const auto tight = tight_solver.solve(source, target, matches, 0.005);
  const auto wide = solver.solve(source, target, matches, 0.2);
  ASSERT_TRUE(tight.valid);
  ASSERT_TRUE(wide.valid);
  ASSERT_GT((tight.translation - wide.translation).norm(), 1);
  const std::vector<double> bounds{0.005, 0.2};
  Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, matches.size()), dst(3, matches.size());
  for (std::size_t i = 0; i < matches.size(); ++i) {
    const auto& p = source[matches[i].first];
    const auto& q = target[matches[i].second];
    src.col(i) << p.x, p.y, p.z;
    dst.col(i) << q.x, q.y, q.z;
  }
  const auto verify_src = teaser::test::teaserPointCloudToEigenMatrix<double>(source);
  const auto verify_dst = teaser::test::teaserPointCloudToEigenMatrix<double>(target);
  const auto indexed = solver.solve(source, target, matches, bounds);
  const auto matched = solver.solve(src, dst, bounds);
  ASSERT_TRUE(indexed.valid);
  ASSERT_TRUE(matched.valid);
  EXPECT_EQ(indexed.rotation, matched.rotation);
  EXPECT_EQ(indexed.translation, matched.translation);

  const auto selected = solver.solve(src, dst, bounds, verify_src, verify_dst);
  ASSERT_TRUE(selected.valid);
  EXPECT_DOUBLE_EQ(solver.getParams().noise_bound, 0.005);
  EXPECT_TRUE(selected.rotation.isApprox(tight.rotation));
  EXPECT_TRUE(selected.translation.isApprox(tight.translation));
  EXPECT_TRUE(solver.getSolution().translation.isApprox(tight.translation));
  EXPECT_EQ(solver.getScaleInliersMask(), tight_solver.getScaleInliersMask());
  EXPECT_EQ(solver.getInlierMaxClique(), tight_solver.getInlierMaxClique());
  EXPECT_EQ(solver.getInlierGraph(), tight_solver.getInlierGraph());
  EXPECT_EQ(solver.getSrcTIMs(), tight_solver.getSrcTIMs());
  EXPECT_EQ(solver.getDstTIMs(), tight_solver.getDstTIMs());
  EXPECT_EQ(solver.getSrcTIMsMap(), tight_solver.getSrcTIMsMap());
  EXPECT_EQ(solver.getDstTIMsMap(), tight_solver.getDstTIMsMap());
  const auto descending =
      solver.solve(src, dst, std::vector<double>{0.2, 0.005}, verify_src, verify_dst);
  EXPECT_EQ(descending.rotation, selected.rotation);
  EXPECT_EQ(descending.translation, selected.translation);
  EXPECT_DOUBLE_EQ(solver.getParams().noise_bound, 0.005);

  const Eigen::Matrix<double, 3, Eigen::Dynamic> empty(3, 0);
  EXPECT_THROW(solver.solve(src, dst, bounds, verify_src, empty), std::invalid_argument);
  const auto single = solver.solve(src, dst, std::vector<double>{0.005}, empty, empty);
  ASSERT_TRUE(single.valid);
  EXPECT_EQ(single.rotation, tight.rotation);
  EXPECT_EQ(single.translation, tight.translation);
}

TEST(RegistrationTest, MultipleNoiseBoundsPreserveInclusiveCompatibilityThreshold) {
  using Solver = teaser::RobustRegistrationSolver;
  Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, 5);
  src << 0, 0, 0, -4, 1,
         0, 4, 0, 0, 0,
         0, 0, 4, 0, 0;
  Eigen::Matrix<double, 3, Eigen::Dynamic> dst = src.colwise() + Eigen::Vector3d(3, -2, 1);
  dst(0, 4) += 1;
  Solver::Params params;
  params.estimate_scaling = false;
  params.cbar2 = 4;
  params.max_clique_num_threads = 1;
  Solver solver(params);
  ASSERT_TRUE(solver.solve(src, dst, std::vector<double>{0.25, 0.125, 0.25}).valid);
  Solver independent(params);
  const auto selected_bound = solver.getParams().noise_bound;
  ASSERT_TRUE(independent.solve(src, dst, selected_bound).valid);
  EXPECT_EQ(solver.getScaleInliersMask(), independent.getScaleInliersMask());
  EXPECT_EQ(solver.getInlierGraph(), independent.getInlierGraph());
  EXPECT_EQ(solver.getInlierMaxClique(), independent.getInlierMaxClique());
  // Pair (0, 4) has length error 1, exactly the threshold for bound 0.25.
  EXPECT_EQ(solver.getSrcTIMsMap().col(3), Eigen::Vector2i(0, 4));
  EXPECT_EQ(solver.getScaleInliersMask()(3), selected_bound == 0.25);
}

TEST(RegistrationTest, PreviousCliqueDoesNotReplaceTheFreshTiedClique) {
  using Solver = teaser::RobustRegistrationSolver;
  const std::vector<teaser::PointXYZ> tetrahedron{{0, 0, 0}, {1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
  teaser::PointCloud source, target;
  std::vector<std::pair<int, int>> matches;
  for (const auto& p : tetrahedron) {
    source.push_back(p);
    target.push_back(p);
  }
  for (const auto& p : tetrahedron) {
    source.push_back({20 + p.x, p.y, p.z});
    target.push_back({40 + 1.05f * p.x, 1.05f * p.y, 1.05f * p.z});
  }
  for (int i = 0; i < 8; ++i) {
    matches.emplace_back(i, i);
  }
  // The full clouds favor the pose from the second four-point clique.
  for (int i = 0; i < 20; ++i) {
    const teaser::PointXYZ p{19.f + 0.15f * i, 0.3f * (i % 5), 0.45f * (i / 4)};
    source.push_back(p);
    target.push_back({p.x + 20.0125f, p.y + 0.025f, p.z + 0.0375f});
  }
  Solver::Params params;
  params.estimate_scaling = false;
  params.max_clique_num_threads = 1;
  Solver narrow(params), wide(params), sweep(params);
  ASSERT_TRUE(narrow.solve(source, target, matches, 0.01).valid);
  const auto expected = wide.solve(source, target, matches, 0.2);
  ASSERT_TRUE(expected.valid);
  ASSERT_EQ(narrow.getInlierMaxClique().size(), 4);
  ASSERT_EQ(wide.getInlierMaxClique().size(), 4);
  ASSERT_NE(narrow.getInlierMaxClique(), wide.getInlierMaxClique());
  const auto verify_src = teaser::test::teaserPointCloudToEigenMatrix<double>(source);
  const auto verify_dst = teaser::test::teaserPointCloudToEigenMatrix<double>(target);
  const auto selected = sweep.solve(verify_src.leftCols(8), verify_dst.leftCols(8),
                                   std::vector<double>{0.01, 0.2}, verify_src, verify_dst);
  ASSERT_TRUE(selected.valid);
  EXPECT_DOUBLE_EQ(sweep.getParams().noise_bound, 0.2);
  EXPECT_EQ(sweep.getInlierMaxClique(), wide.getInlierMaxClique());
  EXPECT_EQ(selected.rotation, expected.rotation);
  EXPECT_EQ(selected.translation, expected.translation);
}

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
