/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#pragma once

#include <vector>

namespace teaser {

struct PointXYZ {
  float x;
  float y;
  float z;

  friend inline bool operator==(const PointXYZ& lhs, const PointXYZ& rhs) {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
  }
  friend inline bool operator!=(const PointXYZ& lhs, const PointXYZ& rhs) { return !(lhs == rhs); }
};

class PointCloud {
public:
  /**
   * @brief Default constructor for PointCloud
   */
  PointCloud() = default;

  // c++ container named requirements
  using value_type = PointXYZ;
  using reference = PointXYZ&;
  using const_reference = const PointXYZ&;
  using difference_type = std::vector<PointXYZ>::difference_type;
  using size_type = std::vector<PointXYZ>::size_type;

  // iterators
  using iterator = std::vector<PointXYZ>::iterator;
  using const_iterator = std::vector<PointXYZ>::const_iterator;
  inline iterator begin() { return points_.begin(); }
  inline iterator end() { return points_.end(); }
  inline const_iterator begin() const { return points_.begin(); }
  inline const_iterator end() const { return points_.end(); }

  // capacity
  inline size_t size() const { return points_.size(); }
  inline void reserve(size_t n) { points_.reserve(n); }
  inline bool empty() { return points_.empty(); }

  // element access
  inline PointXYZ& operator[](size_t i) { return points_[i]; }
  inline const PointXYZ& operator[](size_t i) const { return points_[i]; }
  inline PointXYZ& at(size_t n) { return points_.at(n); }
  inline const PointXYZ& at(size_t n) const { return points_.at(n); }
  inline PointXYZ& front() { return points_.front(); }
  inline const PointXYZ& front() const { return points_.front(); }
  inline PointXYZ& back() { return points_.back(); }
  inline const PointXYZ& back() const { return points_.back(); }

  inline void push_back(const PointXYZ& pt) { points_.push_back(pt); }
  inline void push_back(PointXYZ& pt) { points_.push_back(pt); }

  inline void clear() { points_.clear(); }

private:
  std::vector<PointXYZ> points_;
};


} // namespace teaser
