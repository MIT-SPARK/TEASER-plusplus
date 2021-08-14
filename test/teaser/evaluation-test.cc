/**
 * Copyright 2021, Massachusetts Institute of Technology,
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
#include "teaser/evaluation.h"
#include "teaser/ply_io.h"
#include "test_utils.h"

TEST(EvaluationTest, Simple) {
  // Random small point cloud
  int N = 10;
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

  // Initialize the evaluator
  teaser::SolutionEvaluator evaluator(src, tgt);
  auto err = evaluator.computeErrorMetric(T.topLeftCorner(3, 3), T.topRightCorner(3, 1));
  EXPECT_NEAR(err, 0, 1e-6);
}

