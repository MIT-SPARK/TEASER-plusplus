/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"

#include <iostream>

#include "teaser/ply_io.h"

TEST(IOTest, ImportPLY) {
  teaser::PLYReader reader;
  teaser::PointCloud cloud;
  auto status = reader.read("./data/cube.ply", cloud);
  EXPECT_EQ(status, 0);

  teaser::PLYWriter writer;
  const std::string test_file_name = "IOTest_ImportPLY.ply";
  status = writer.write(test_file_name, cloud);
  EXPECT_EQ(status, 0);

  teaser::PointCloud cloud2;
  status = reader.read(test_file_name, cloud2);
  EXPECT_EQ(status, 0);

  // Compare two clouds
  ASSERT_EQ(cloud.size(), cloud2.size());

  for (size_t i = 1; i < cloud.size(); ++i) {
    ASSERT_EQ(cloud[i].x, cloud2[i].x);
    ASSERT_EQ(cloud[i].y, cloud2[i].y);
    ASSERT_EQ(cloud[i].z, cloud2[i].z);
  }
}

TEST(IOTest, ImportBigPLY) {
  teaser::PLYReader reader;
  teaser::PointCloud cloud;
  auto status = reader.read("./data/uw-rgbdv2-01.ply", cloud);
  EXPECT_EQ(status, 0);
}
