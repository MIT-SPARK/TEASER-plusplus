/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <flann/flann.hpp>

#include <pcl/features/fpfh.h>

#include "teaser/geometry.h"
#include "fpfh.h"

namespace teaser {

class Matcher {
public:
  typedef std::vector<Eigen::VectorXf> Feature;
  typedef flann::Index<flann::L2<float>> KDTree;

  // New methods
  // Public methods:
  // 1. calculateCorrespondences
  //    input: source point cloud, target point cloud
  //    output: correspondences
  Matcher() = default;

  /**
   * Calculate correspondences based on given features and point clouds.
   * @param source_points
   * @param target_points
   * @param use_absolute_scale
   * @param use_crosscheck
   * @param use_tuple_test
   * @return
   */
  std::vector<std::pair<int, int>>
  calculateCorrespondences(teaser::PointCloud& source_points, teaser::PointCloud& target_points,
                           teaser::FPFHCloud& source_features, teaser::FPFHCloud& target_features,
                           bool use_absolute_scale = true, bool use_crosscheck = true,
                           bool use_tuple_test = true, float tuple_scale = 0);
  /**
   * Calculate correspondences based on given features and point clouds.
   * @param source_points Pointer to a pcl::PointCloud<pcl::PointXYZ>
   * @param target_points Pointer to a pcl::PointCloud<pcl::PointXYZ>
   * @param source_features Pointer to a pcl::PointCloud<pcl::FPFHSignature33>
   * @param target_features Pointer to a pcl::PointCloud<pcl::FPFHSignature33>
   * @param use_absolute_scale
   * @param use_crosscheck
   * @param use_tuple_test
   * @return
   */
  std::vector<std::pair<int, int>>
  calculateCorrespondences(pcl::PointCloud<pcl::PointXYZINormal>::Ptr source_points,
                           pcl::PointCloud<pcl::PointXYZINormal>::Ptr target_points,
                           pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features,
                           pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features,
                           bool use_absolute_scale = true, bool use_crosscheck = true,
                           bool use_tuple_test = true, float tuple_scale = 0);
  /**
   * @brief For all source points, find top k matching correspondences.
   * @param source_points
   * @param target_points
   * @param source_features
   * @param target_features
   * @param k
   * @return
   */
  std::vector<std::pair<int, int>>
  calculateKCorrespondences(teaser::PointCloud& source_points, teaser::PointCloud& target_points,
                            teaser::FPFHCloud& source_features, teaser::FPFHCloud& target_features,
                            int k = 10);

  /**
   * @brief For all source points, find top k matching correspondences.
   * @param source_points
   * @param target_points
   * @param source_features
   * @param target_features
   * @param k
   * @return
   */
  std::vector<std::pair<int, int>>
  calculateKCorrespondences(pcl::PointCloud<pcl::PointXYZINormal>::Ptr source_points,
                            pcl::PointCloud<pcl::PointXYZINormal>::Ptr target_points,
                            pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features,
                            pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features,
                            int k = 10);

private:
  template <typename T> void buildKDTree(const std::vector<T>& data, KDTree* tree);

  template <typename T>
  void searchKDTree(KDTree* tree, const T& input, std::vector<int>& indices,
                    std::vector<float>& dists, int nn);

  void kMatching(int k);

  void advancedMatching(bool use_crosscheck, bool use_tuple_test, float tuple_scale);

  void normalizePoints(bool use_absolute_scale);

  std::vector<std::pair<int, int>> corres_;
  std::vector<teaser::PointCloud> pointcloud_;
  std::vector<Feature> features_;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > means_; // for normalization
  float global_scale_;
};

} // namespace teaser
