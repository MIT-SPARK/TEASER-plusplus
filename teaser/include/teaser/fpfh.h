/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <boost/smart_ptr/shared_ptr.hpp>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>

#include "teaser/geometry.h"

namespace teaser {

using FPFHCloud = pcl::PointCloud<pcl::FPFHSignature33>;
using FPFHCloudPtr = pcl::PointCloud<pcl::FPFHSignature33>::Ptr;

class FPFHEstimation {
public:
  FPFHEstimation() {
    fpfh_estimation_.reset(
        new pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>());
    normals_.reset(new pcl::PointCloud<pcl::Normal>());
  }
  /**
   * Compute FPFH features.
   *
   * @return A shared pointer to the FPFH feature point cloud
   * @param input_cloud
   * @param normal_search_radius Radius for estimating normals
   * @param fpfh_search_radius Radius for calculating FPFH (needs to be at least normalSearchRadius)
   */
  FPFHCloudPtr computeFPFHFeatures(const PointCloud& input_cloud,
                                   double normal_search_radius = 0.03,
                                   double fpfh_search_radius = 0.05);

  /**
   * Return the pointer to the underlying pcl::FPFHEstimation object
   * @return
   */
  inline pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr
  getImplPointer() const {
    return fpfh_estimation_;
  }

   /**
   * Return the normal vectors of the input cloud that are used in the calculation of FPFH
   * @return
   */
  inline pcl::PointCloud<pcl::Normal> getNormals() { return *normals_; }

private:
  // pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr fpfh_estimation_;
  pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr fpfh_estimation_;

  pcl::PointCloud<pcl::Normal>::Ptr normals_;

  /**
   * Wrapper function for the corresponding PCL function.
   * @param output_cloud
   */
  void compute(pcl::PointCloud<pcl::FPFHSignature33>& output_cloud);

  /**
   * Wrapper function for the corresponding PCL function.
   * @param input_cloud
   */
  void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud);

  /**
   * Wrapper function for the corresponding PCL function.
   * @param input_normals
   */
  void setInputNormals(pcl::PointCloud<pcl::Normal>::Ptr input_normals);

  /**
   * Wrapper function for the corresponding PCL function.
   * @param search_method
   */
  void setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method);

  /**
   * Wrapper function for the corresponding PCL function.
   */
  void setRadiusSearch(double);
};

} // namespace teaser
