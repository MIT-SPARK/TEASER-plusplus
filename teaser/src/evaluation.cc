/**
 * Copyright 2021, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <teaser/evaluation.h>

teaser::SolutionEvaluator::SolutionEvaluator(const Eigen::Matrix3d& src, const Eigen::Matrix3d& dst,
                                             double corr_dist_threshold)
    : src_(src), dst_(dst), corr_dist_threshold_(corr_dist_threshold) {
  buildKDTree(dst_);
  error_functor_ = new TruncatedError(corr_dist_threshold);
}

void teaser::SolutionEvaluator::setPointClouds(const Eigen::Matrix3d& src,
                                               const Eigen::Matrix3d& dst) {
  src_ = src;
  dst_ = dst;
  buildKDTree(dst_);
}

void teaser::SolutionEvaluator::buildKDTree(const Eigen::Matrix3d& dst) {
  int num_pts, dim;
  num_pts = static_cast<int>(dst.cols());
  dim = static_cast<int>(dst.rows());
  std::vector<double> dataset(num_pts * dim);
  flann::Matrix<double> dataset_mat(&dataset[0], num_pts, dim);
  for (int i = 0; i < num_pts; i++) {
    for (int j = 0; j < dim; j++) {
      dataset[i * dim + j] = dst(j, i);
    }
  }
  tree_ = new KDTree(dataset_mat, flann::KDTreeSingleIndexParams(15));
  tree_->buildIndex();
}

double teaser::SolutionEvaluator::computeErrorMetric(const Eigen::Matrix3d& rotation,
                                                     const Eigen::Vector3d& translation) {
  // apply transformation to src
  Eigen::Matrix3d transformed_src = rotation * src_;
  transformed_src.colwise() += translation;
  std::vector<int> corres_K;
  std::vector<double> dis;
  std::vector<std::pair<int, int>> corres;
  double error = 0;
  for (size_t i = 0; i < src_.cols(); ++i) {
    // find the distance between the point and its nearest neighbor in the target point cloud
    Eigen::Vector3d pt = transformed_src.col(i);
    searchKDTree(tree_, pt, corres_K, dis, 1);

    // compute errors
    error += (*error_functor_)(dis[0]);
  }
  return error;
}

template <typename T>
void teaser::SolutionEvaluator::searchKDTree(teaser::SolutionEvaluator::KDTree* tree,
                                             const T& input, std::vector<int>& indices,
                                             std::vector<double>& dists, int nn) {
  int rows_t = 1;
  int dim = input.size();

  std::vector<double> query;
  query.resize(rows_t * dim);
  for (int i = 0; i < dim; i++)
    query[i] = input(i);
  flann::Matrix<double> query_mat(&query[0], rows_t, dim);

  indices.resize(rows_t * nn);
  dists.resize(rows_t * nn);
  flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
  flann::Matrix<double> dists_mat(&dists[0], rows_t, nn);

  tree->knnSearch(query_mat, indices_mat, dists_mat, nn, flann::SearchParams(128));
}

void teaser::SolutionEvaluator::clear() {
  delete tree_;
  tree_ = nullptr;
}

teaser::SolutionEvaluator::~SolutionEvaluator() { delete tree_; }
