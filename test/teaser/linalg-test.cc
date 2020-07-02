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

#include <Eigen/Core>

#include "teaser/linalg.h"

TEST(LinalgTest, HatMap) {
  {
    // all zeros
    Eigen::Matrix<double, 3, 1> v1;
    v1 << 0,0,0;
    Eigen::Matrix3d exp;
    exp.setZero();
    const auto act = teaser::hatmap(v1);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // non-zero cases
    Eigen::Matrix<double, 3, 1> v1;
    v1 << 1,2,3;
    Eigen::Matrix3d exp;
    // clang-format off
    exp << 0, -3, 2,
           3,  0,-1,
          -2,  1, 0;
    // clang-format on
    const auto act = teaser::hatmap(v1);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // negative
    Eigen::Matrix<double, 3, 1> v1;
    v1 << 0,2,-3;
    Eigen::Matrix3d exp;
    // clang-format off
    exp << 0,   3, 2,
           -3,  0, 0,
           -2,  0, 0;
    // clang-format on
    const auto act = teaser::hatmap(v1);
    ASSERT_TRUE(act.isApprox(exp));
  }
}

TEST(LinalgTest, VectorKronFixedSize) {
  {
    // a simple case
    Eigen::Matrix<double, 3, 1> v1;
    v1 << 1,1,1;
    Eigen::Matrix<double, 3, 1> v2;
    v2 << 1,1,1;
    Eigen::Matrix<double, 9, 1> exp = Eigen::Matrix<double, 9, 1>::Ones(9,1);

    // compute the actual result
    Eigen::Matrix<double, 9, 1> act;
    teaser::vectorKron<double, 3, 3>(v1, v2, &act);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // non-zero vectors
    Eigen::Matrix<double, 2, 1> v1;
    v1 << 1,4;
    Eigen::Matrix<double, 2, 1> v2;
    v2 << 3,2;
    Eigen::Matrix<double, 4, 1> exp;
    exp << 3,2,12,8;

    // compute the actual result
    Eigen::Matrix<double, 4, 1> act;
    teaser::vectorKron<double, 2, 2>(v1, v2, &act);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // different lengths
    Eigen::Matrix<double, 3, 1> v1;
    v1 << 1,-3,3;
    Eigen::Matrix<double, 4, 1> v2;
    v2 << 3,-1,9,-10;
    Eigen::Matrix<double, 12, 1> exp;
    exp << 3,-1,9,-10,-9,3,-27,30,9,-3,27,-30;

    // compute the actual result
    Eigen::Matrix<double, 12, 1> act;
    teaser::vectorKron<double, 3, 4>(v1, v2, &act);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // different lengths (all zero)
    Eigen::Matrix<double, 3, 1> v1;
    v1 << 0,0,0;
    Eigen::Matrix<double, 4, 1> v2;
    v2 << 3,-1,9,-10;
    Eigen::Matrix<double, 12, 1> exp;
    exp.setZero();

    // compute the actual result
    Eigen::Matrix<double, 12, 1> act;
    teaser::vectorKron<double, 3, 4>(v1, v2, &act);
    ASSERT_TRUE(act.isApprox(exp));
  }
}

TEST(LinalgTest, VectorKronDynamicSize) {
  {
    // a simple case
    Eigen::VectorXd v1(3,1);
    v1 << 1,1,1;
    Eigen::VectorXd v2(3,1);
    v2 << 1,1,1;
    Eigen::VectorXd exp(9,1);
    exp.setOnes();

    // compute the actual result
    Eigen::VectorXd act = teaser::vectorKron<double, 3, 3>(v1, v2);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // non-zero vectors
    Eigen::VectorXd v1(2,1);
    v1 << 1,4;
    Eigen::VectorXd v2(2,1);
    v2 << 3,2;
    Eigen::VectorXd exp(4,1);
    exp << 3,2,12,8;

    // compute the actual result
    Eigen::VectorXd act = teaser::vectorKron<double, 2, 2>(v1, v2);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // different lengths
    Eigen::VectorXd v1(3,1);
    v1 << 1,-3,3;
    Eigen::VectorXd v2(4,1);
    v2 << 3,-1,9,-10;
    Eigen::VectorXd exp(12,1);
    exp << 3,-1,9,-10,-9,3,-27,30,9,-3,27,-30;

    // compute the actual result
    Eigen::VectorXd act = teaser::vectorKron<double, 3, 4>(v1, v2);
    ASSERT_TRUE(act.isApprox(exp));
  }
  {
    // different lengths (all zero)
    Eigen::VectorXd v1(3,1);
    v1 << 0,0,0;
    Eigen::VectorXd v2(4,1);
    v2 << 3,-1,9,-10;
    Eigen::VectorXd exp(12,1);
    exp.setZero();

    // compute the actual result
    Eigen::VectorXd act = teaser::vectorKron<double, 3, 4>(v1, v2);
    ASSERT_TRUE(act.isApprox(exp));
  }
}
