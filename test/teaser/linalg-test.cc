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

TEST(LinalgTest, VectorKron) {
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

}
