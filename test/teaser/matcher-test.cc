/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"

#include <iostream>
#include <fstream>

#include <teaser/matcher.h>
#include <teaser/ply_io.h>
#include "test_utils.h"

TEST(FPFHMatcherTest, SelfMatching) {
  teaser::PLYReader reader;
  teaser::PointCloud cloud1;
  teaser::PointCloud cloud2;
  auto status = reader.read("./data/canstick.ply", cloud1);
  EXPECT_EQ(status, 0);
  status = reader.read("./data/canstick.ply", cloud2);
  EXPECT_EQ(status, 0);

  teaser::FPFHEstimation fpfh;
  auto descriptors1 = fpfh.computeFPFHFeatures(cloud1, 0.03, 0.05);
  auto descriptors2 = fpfh.computeFPFHFeatures(cloud2, 0.03, 0.05);

  teaser::Matcher matcher;
  auto correspondences = matcher.calculateCorrespondences(cloud1, cloud2, *descriptors1,
                                                          *descriptors2, false, true, false, 0);
  EXPECT_EQ(correspondences.size(), cloud1.size());

  for (auto& pair : correspondences) {
    EXPECT_EQ(pair.first, pair.second);
  }
}

TEST(FPFHMatcherTest, MatchCase1) {
  teaser::PLYReader reader;
  teaser::PointCloud obj_cloud;
  teaser::PointCloud scene_cloud;
  auto status = reader.read("./data/matcher-test-object-1.ply", obj_cloud);
  EXPECT_EQ(status, 0);
  status = reader.read("./data/matcher-test-scene-1.ply", scene_cloud);
  EXPECT_EQ(status, 0);

  teaser::FPFHEstimation fpfh;
  auto obj_descriptors = fpfh.computeFPFHFeatures(obj_cloud, 0.02, 0.04);
  auto scene_descriptors = fpfh.computeFPFHFeatures(scene_cloud, 0.02, 0.04);

  teaser::Matcher matcher;
  auto correspondences = matcher.calculateCorrespondences(
      obj_cloud, scene_cloud, *obj_descriptors, *scene_descriptors, false, true, false, 0.95);

  // load reference matches/correspondences
  std::ifstream file("./data/matcher-test-matches-1.csv");

  std::vector<std::pair<int, int>> ref_correspondences;
  while (true) {
    auto tokens = teaser::test::getNextLineAndSplitIntoTokens(file);
    if (tokens.size() <= 1) {
      break;
    }
    ASSERT_EQ(tokens.size(), 2);
    ref_correspondences.emplace_back(
        // -1 because the ref correspondences use 1-index (MATLAB)
        std::pair<int, int>{std::stoi(tokens[0]) - 1, std::stoi(tokens[1]) - 1});
  }

  // compare calculated correspondences with reference correspondences
  for (size_t i = 0; i < ref_correspondences.size(); ++i) {
    EXPECT_EQ(correspondences[i].first, ref_correspondences[i].first);
    EXPECT_EQ(correspondences[i].second, ref_correspondences[i].second);
  }
}

TEST(FPFHMatcherTest, kNeighborMatchSingle) {
  teaser::PLYReader reader;
  teaser::PointCloud obj_cloud;
  teaser::PointCloud scene_cloud;
  auto status_0 = reader.read("./data/matcher-test-object-1.ply", obj_cloud);
  EXPECT_EQ(status_0, 0);
  auto status_1 = reader.read("./data/matcher-test-object-1.ply", scene_cloud);
  EXPECT_EQ(status_1, 0);

  teaser::FPFHEstimation fpfh;
  auto obj_descriptors = fpfh.computeFPFHFeatures(obj_cloud, 0.02, 0.04);
  auto scene_descriptors = fpfh.computeFPFHFeatures(scene_cloud, 0.02, 0.04);

  teaser::Matcher matcher;
  auto correspondences = matcher.calculateKCorrespondences(obj_cloud, scene_cloud, *obj_descriptors,
                                                           *scene_descriptors, 1);

  for (size_t i = 0; i < correspondences.size(); ++i) {
    EXPECT_EQ(correspondences[i].first, i);
    EXPECT_EQ(correspondences[i].second, i);
  }
 }


TEST(FPFHMatcherTest, kNeighborMatchMultiple) {
  teaser::PLYReader reader;
  teaser::PointCloud obj_cloud;
  teaser::PointCloud scene_cloud;
  auto status_0 = reader.read("./data/matcher-test-object-1.ply", obj_cloud);
  EXPECT_EQ(status_0, 0);
  auto status_1 = reader.read("./data/matcher-test-object-1.ply", scene_cloud);
  EXPECT_EQ(status_1, 0);

  teaser::FPFHEstimation fpfh;
  auto obj_descriptors = fpfh.computeFPFHFeatures(obj_cloud, 0.02, 0.04);
  auto scene_descriptors = fpfh.computeFPFHFeatures(scene_cloud, 0.02, 0.04);

  int og_size = obj_cloud.size();
  int nn = 5;
  teaser::Matcher matcher;
  auto correspondences = matcher.calculateKCorrespondences(obj_cloud, scene_cloud, *obj_descriptors,
                                                           *scene_descriptors, nn);
  EXPECT_EQ(correspondences.size(), nn * og_size);
  for (int i = 0; i < correspondences.size(); ++i) {
    int pt_idx = i / nn;
    int k_idx = i % nn;
    EXPECT_EQ(correspondences[i].first, pt_idx);
    if (k_idx == 0) {
      EXPECT_EQ(correspondences[i].second, pt_idx);
    }
  }
}