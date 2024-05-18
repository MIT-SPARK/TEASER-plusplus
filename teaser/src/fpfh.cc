/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "teaser/fpfh.h"

#include <pcl/features/normal_3d.h>

#include "teaser/utils.h"

teaser::FPFHCloudPtr teaser::FPFHEstimation::computeFPFHFeatures(
    const teaser::PointCloud& input_cloud, double normal_search_radius, double fpfh_search_radius) {

  // Intermediate variables
  teaser::FPFHCloudPtr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto& i : input_cloud) {
    pcl::PointXYZ p(i.x, i.y, i.z);
    pcl_input_cloud->push_back(p);
  }

  // Estimate normals
  normals_->clear();
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
  normalEstimation.setInputCloud(pcl_input_cloud);
  normalEstimation.setRadiusSearch(normal_search_radius);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  normalEstimation.setSearchMethod(kdtree);
  normalEstimation.compute(*normals_);

  // Estimate FPFH
  setInputCloud(pcl_input_cloud);
  setInputNormals(normals_);
  setSearchMethod(kdtree);
  setRadiusSearch(fpfh_search_radius);
  compute(*descriptors);

  return descriptors;
}

void teaser::FPFHEstimation::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud) {
  fpfh_estimation_->setInputCloud(input_cloud);
}

void teaser::FPFHEstimation::setInputNormals(pcl::PointCloud<pcl::Normal>::Ptr input_normals) {
  fpfh_estimation_->setInputNormals(input_normals);
}

void teaser::FPFHEstimation::setSearchMethod(
    pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method) {
  fpfh_estimation_->setSearchMethod(search_method);
}

void teaser::FPFHEstimation::compute(pcl::PointCloud<pcl::FPFHSignature33>& output_cloud) {
  fpfh_estimation_->compute(output_cloud);
}

void teaser::FPFHEstimation::setRadiusSearch(double r) { fpfh_estimation_->setRadiusSearch(r); }
