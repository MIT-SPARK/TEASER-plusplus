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
#include <random>
#include "teaser/utils.h"

TEST(UtilsTest, RandomSample) {
  std::vector<int> v1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> v1_ref(v1);

  std::random_device rd;
  std::mt19937 g(rd());
  size_t num_samples = 5;
  auto v2 = teaser::utils::randomSample(v1, num_samples, g);

  // Make sure the output vector has the correct size.
  ASSERT_EQ(v2.size(), num_samples);

  // Make sure the input vector has not been changed.
  EXPECT_THAT(v1, ::testing::Eq(v1_ref));

  // Make sure the sample vector has all unique elements.
  auto it = std::unique(v2.begin(), v2.end());
  EXPECT_TRUE(it == v2.end());

  // Make sure all elements in the sample vector are in the original vector.
  for (auto x : v2) {
    auto in_input_vec = std::find(v1.begin(), v1.end(), x) != v1.end();
    EXPECT_TRUE(in_input_vec);
  }
}

TEST(UtilsTest, EigenMatrixRemoveRowColumn) {
  // remove one column
  {
    Eigen::Matrix<float, 3, Eigen::Dynamic> v1(3, 4);
    // clang-format off
    v1 << 0, 1, 1, 1,
          0, 1, 1, 1,
          0, 1, 1, 1;
    // clang-format on
    Eigen::Matrix<float, 3, Eigen::Dynamic> v1_after(3, 3);
    // clang-format off
    v1_after << 1, 1, 1,
                1, 1, 1,
                1, 1, 1;
    // clang-format on
    teaser::utils::removeColumn<float, 3, Eigen::Dynamic>(v1, 0);
    EXPECT_EQ(v1.cols(), 3);
    EXPECT_EQ(v1, v1_after);
  }
  // remove one row
  {
    Eigen::Matrix<float, Eigen::Dynamic, 4> v1(3, 4);
    // clang-format off
    v1 << 1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12;
    // clang-format on
    Eigen::Matrix<float, Eigen::Dynamic, 4> v1_after(2, 4);
    // clang-format off
    v1_after << 1, 2, 3, 4,
                5, 6, 7, 8;
    // clang-format on
    teaser::utils::removeRow<float, Eigen::Dynamic, 4>(v1, 2);
    EXPECT_EQ(v1.rows(), 2);
    EXPECT_EQ(v1, v1_after);
  }
  // remove one row and column
  {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v1(3, 4);
    // clang-format off
    v1 << 1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12;
    // clang-format on
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v1_after(2, 3);
    // clang-format off
    v1_after << 1, 2, 3,
                5, 6, 7;
    // clang-format on
    teaser::utils::removeRow<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 2);
    teaser::utils::removeColumn<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 3);
    EXPECT_EQ(v1, v1_after);
  }
  // remove all
  {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> v1(3, 4);
    // clang-format off
    v1 << 1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12;
    // clang-format on
    teaser::utils::removeRow<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 1);
    teaser::utils::removeRow<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 1);
    teaser::utils::removeRow<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 0);
    teaser::utils::removeColumn<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 3);
    teaser::utils::removeColumn<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 1);
    teaser::utils::removeColumn<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 1);
    teaser::utils::removeColumn<float, Eigen::Dynamic, Eigen::Dynamic>(v1, 0);
    EXPECT_EQ(v1.rows(), 0);
    EXPECT_EQ(v1.cols(), 0);
  }
}

TEST(UtilsTest, CalculatePointClusterDiameter) {
  {
    Eigen::Matrix<float, 3, Eigen::Dynamic> test_mat(3,3);
    test_mat << -1, 0, 1,
                -1, 0, 1,
                -1, 0, 1;
    float d = teaser::utils::calculateDiameter<float, 3>(test_mat);
    EXPECT_NEAR(d, 3.4641, 0.0001);
  }
  {
    Eigen::Matrix<float, 3, Eigen::Dynamic> test_mat(3,4);
    test_mat << 1, 2, 3, 4,
                1, 2, 3, 4,
                1, 2, 3, 4;
    float d = teaser::utils::calculateDiameter<float, 3>(test_mat);
    EXPECT_NEAR(d, 5.1962, 0.0001);
  }
}
