/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <teaser/fpfh.h>

TEST(FPFHTest, CalculateFPFHFeaturesWithPCL) {

  // Object for storing the point cloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // Object for storing the normals.
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  // Object for storing the FPFH descriptors for each point.
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(
      new pcl::PointCloud<pcl::FPFHSignature33>());

  // Read a PCD file from disk.
  pcl::PCLPointCloud2 cloud_blob;
  pcl::io::loadPCDFile("./data/bunny.pcd", cloud_blob);
  pcl::fromPCLPointCloud2(cloud_blob, *cloud);

  // Estimate the normals.
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
  normalEstimation.setInputCloud(cloud);
  normalEstimation.setRadiusSearch(0.03);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  normalEstimation.setSearchMethod(kdtree);
  normalEstimation.compute(*normals);

  // Estimate FPFH features.
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
  fpfh.setInputCloud(cloud);
  fpfh.setInputNormals(normals);
  fpfh.setSearchMethod(kdtree);
  fpfh.setRadiusSearch(0.05);
  fpfh.compute(*descriptors);

  ASSERT_EQ(cloud->size(), descriptors->size());
}

TEST(FPFHTest, CalculateFPFHFeaturesWithTeaserInterface) {
  teaser::FPFHEstimation fpfh;

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2 cloud_blob;
  pcl::io::loadPCDFile("./data/bunny.pcd", cloud_blob);
  pcl::fromPCLPointCloud2(cloud_blob, *pcl_cloud);
  teaser::PointCloud input_cloud;
  for (auto& p : *pcl_cloud) {
    input_cloud.push_back({p.x, p.y, p.z});
  }

  auto descriptors = fpfh.computeFPFHFeatures(input_cloud, 0.03, 0.05);

  ASSERT_EQ(pcl_cloud->size(), descriptors->size());

  std::ifstream file("./data/bunny_fpfh.csv");
  if (!file.good()) {
    FAIL() << "Error opening test file. Check whether CMake has copied the file to the binary "
              "directory.";
  }
  std::string str;
  std::vector<float> ref_vals;
  while (std::getline(file, str)) {
    ref_vals.push_back(std::stof(str));
  }

  int count = 0;
  for (auto& f : *descriptors) {
    for (int i = 0; i < 33; ++i) {
      auto val = f.histogram[i];
      EXPECT_NEAR(val, ref_vals[count], 1e-4);
      count++;
    }
  }
}
