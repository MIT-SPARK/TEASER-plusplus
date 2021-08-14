/**
 * Copyright 2021, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <flann/flann.hpp>

#include <Eigen/Core>

namespace teaser {

/**
 * @brief A helper class for computing the error metrics of estimated transformations
 */
class SolutionEvaluator {
public:
  class ErrorFunctor {
  public:
    virtual ~ErrorFunctor() = default;
    virtual double operator()(double d) const = 0;
  };

  class HuberPenalty : public ErrorFunctor {
  public:
    HuberPenalty() = delete;
    explicit HuberPenalty(double threshold) : threshold_(threshold) {}
    double operator()(double e) const override {
      if (e <= threshold_) {
        return (0.5 * e * e);
      } else {
        return (0.5 * threshold_ * (2.0 * std::fabs(e) - threshold_));
      }
    }

  protected:
    double threshold_;
  };

  class TruncatedError : public ErrorFunctor {
  public:
    TruncatedError() = delete;
    explicit TruncatedError(double threshold) : threshold_(threshold) {}
    double operator()(double e) const override {
      if (e <= threshold_) {
        return e / threshold_;
      } else {
        return 1.0;
      }
    }

  protected:
    double threshold_;
  };

  SolutionEvaluator() = delete;

  SolutionEvaluator(const Eigen::Matrix3d& src, const Eigen::Matrix3d& dst,
                    double corr_dist_threshold = 100);

  void setPointClouds(const Eigen::Matrix3d& src, const Eigen::Matrix3d& dst);

  double computeErrorMetric(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation);

  void clear();

  ~SolutionEvaluator();

private:
  typedef flann::Index<flann::L2<double>> KDTree;

  void buildKDTree(const Eigen::Matrix3d& dst);

  template <typename T>
  void searchKDTree(KDTree* tree, const T& input, std::vector<int>& indices,
                    std::vector<double>& dists, int nn);

  double corr_dist_threshold_ = 100;
  Eigen::Matrix3d src_;
  Eigen::Matrix3d dst_;
  KDTree* tree_ = nullptr;
  ErrorFunctor* error_functor_ = nullptr;
};

} // namespace teaser
