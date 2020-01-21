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
#include "teaser/geometry.h"

TEST(PointXYZTest, SimpleOperations) {
  teaser::PointXYZ p1{1, 2, 3};
  teaser::PointXYZ p2{1, 2, 3};
  EXPECT_EQ(p1, p2);
  teaser::PointXYZ p3 = p1;
  EXPECT_EQ(p1, p3);
  p1.x = 6;
  EXPECT_NE(p1, p3);
}

TEST(PointCloudTest, SimpleOperations) {
  teaser::PointCloud cloud1;
  EXPECT_TRUE(cloud1.empty());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 10);

  size_t CLOUD_SIZE = 1000;
  cloud1.reserve(CLOUD_SIZE);
  for (size_t i = 0; i < CLOUD_SIZE; ++i) {
    cloud1.push_back({(float)dis(gen), (float)dis(gen), (float)dis(gen)});
  }
  teaser::PointCloud cloud2 = cloud1;
  EXPECT_EQ(cloud1.size(), cloud2.size());
  for (size_t i = 0; i < cloud1.size(); ++i) {
    EXPECT_EQ(cloud1[i].x, cloud2[i].x);
    EXPECT_EQ(cloud1[i].y, cloud2[i].y);
    EXPECT_EQ(cloud1[i].z, cloud2[i].z);
  }

  float SCALE = 10;
  float TRANSLATION = 10;

  for (size_t i = 0; i < cloud1.size(); ++i) {
    cloud1[i].x *= SCALE;
    cloud1[i].y *= SCALE;
    cloud1[i].z *= SCALE;

    EXPECT_EQ(cloud1[i].x, cloud2[i].x * SCALE);
    EXPECT_EQ(cloud1[i].y, cloud2[i].y * SCALE);
    EXPECT_EQ(cloud1[i].z, cloud2[i].z * SCALE);

    cloud1[i].x /= SCALE;
    cloud1[i].y /= SCALE;
    cloud1[i].z /= SCALE;

    EXPECT_EQ(cloud1[i].x, cloud2[i].x);
    EXPECT_EQ(cloud1[i].y, cloud2[i].y);
    EXPECT_EQ(cloud1[i].z, cloud2[i].z);

    cloud1[i].x += TRANSLATION;
    cloud1[i].y += TRANSLATION;
    cloud1[i].z += TRANSLATION;

    EXPECT_EQ(cloud1[i].x, cloud2[i].x + TRANSLATION);
    EXPECT_EQ(cloud1[i].y, cloud2[i].y + TRANSLATION);
    EXPECT_EQ(cloud1[i].z, cloud2[i].z + TRANSLATION);

    cloud1[i].x -= TRANSLATION;
    cloud1[i].y -= TRANSLATION;
    cloud1[i].z -= TRANSLATION;

    EXPECT_EQ(cloud1[i].x, cloud2[i].x);
    EXPECT_EQ(cloud1[i].y, cloud2[i].y);
    EXPECT_EQ(cloud1[i].z, cloud2[i].z);
  }

  // range-based for loops to test iterators
  size_t count = 0;
  for (auto& p : cloud1) {
    count++;
  }
  EXPECT_EQ(count, CLOUD_SIZE);

  cloud1.clear();
  cloud2.clear();
  EXPECT_TRUE(cloud1.empty());
  EXPECT_TRUE(cloud2.empty());
}
