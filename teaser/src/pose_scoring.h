#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace teaser {
namespace detail {

// Rank transforms by geometric agreement relative to the sampling density in each cloud.
class PoseScorer {
public:
  using Points = Eigen::Matrix<double, 3, Eigen::Dynamic>;

  static PoseScorer forMatches(const Points& source, const Points& target) {
    return {IndexedSample(source, source.cols(), source.cols()),
            IndexedSample(target, target.cols(), target.cols())};
  }

  static PoseScorer forVerification(const Points& source, const Points& target) {
    return {verificationIndex(source), verificationIndex(target)};
  }

  static double matchSpacing(const Points& cloud) {
    return std::sqrt(IndexedSample(cloud, cloud.cols(), cloud.cols()).spacingSquared());
  }

  double targetSpacing() const { return std::sqrt(target_.spacingSquared()); }

  double score(double scale, const Eigen::Matrix3d& rotation,
               const Eigen::Vector3d& translation,
               double incumbent = -std::numeric_limits<double>::infinity()) {
    if (scale == previous_scale_ && rotation == previous_rotation_ &&
        translation == previous_translation_) {
      return previous_score_;
    }
    const Eigen::Matrix3d inverse = rotation.transpose() / scale;
    const double forward = directedScore(source_, target_, scale * rotation, translation,
                                         source_neighbors_, 1, incumbent);
    if (0.5 * (forward + 1) < incumbent) {
      return -std::numeric_limits<double>::infinity();
    }
    const double reverse = directedScore(target_, source_, inverse, -inverse * translation,
                                         target_neighbors_, forward, incumbent);
    const double total = 0.5 * (forward + reverse);
    if (std::isfinite(total)) {
      previous_scale_ = scale;
      previous_rotation_ = rotation;
      previous_translation_ = translation;
      previous_score_ = total;
    }
    return total;
  }

private:
  using Point = std::array<double, 3>;
  static constexpr std::size_t kNoPoint = std::numeric_limits<std::size_t>::max();
  static constexpr std::size_t kLeafPoints = 8;

  struct Nearest {
    double distance_squared = std::numeric_limits<double>::infinity();
    std::size_t point = kNoPoint;
  };

  static std::size_t pointCount(const Points& cloud) { return cloud.cols(); }

  static Point pointAt(const Points& cloud, std::size_t i) {
    const auto p = cloud.col(i);
    return {p.x(), p.y(), p.z()};
  }

  static std::size_t pointCount(const std::vector<Point>& cloud) { return cloud.size(); }
  static const Point& pointAt(const std::vector<Point>& cloud, std::size_t i) { return cloud[i]; }

  template <typename Cloud>
  static std::vector<Point> stratifiedSample(const Cloud& cloud, std::size_t limit) {
    const auto size = pointCount(cloud);
    const auto count = std::min(size, limit);
    std::vector<Point> result;
    result.reserve(count);
    if (count == size) {
      for (std::size_t i = 0; i < count; ++i) {
        result.push_back(pointAt(cloud, i));
      }
      return result;
    }
    // Use one deterministic sample per stratum to avoid aliasing scan order.
    std::uint32_t seed = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const auto begin = i * size / count;
      const auto end = (i + 1) * size / count;
      seed = seed * 1664525U + 1013904223U;
      result.push_back(pointAt(cloud, begin + (seed >> 8) % (end - begin)));
    }
    return result;
  }

  class IndexedSample {
  public:
    template <typename Cloud>
    IndexedSample(const Cloud& cloud, std::size_t ref_limit, std::size_t query_limit,
                  double spacing = 0)
        : points_(stratifiedSample(cloud, ref_limit)), query_limit_(query_limit) {
      std::sort(points_.begin(), points_.end());
      points_.erase(std::unique(points_.begin(), points_.end()), points_.end());
      if (points_.size() < 2) {
        throw std::invalid_argument("Pose scoring requires two distinct sample points per cloud");
      }
      build(0, points_.size(), 0);
      if (spacing > 0) {
        spacing_squared_ = spacing * spacing;
        return;
      }

      std::vector<double> distances;
      distances.reserve(queryCount());
      for (std::size_t i = 0; i < queryCount(); ++i) {
        const auto point = queryIndex(i);
        distances.push_back(std::sqrt(nearest(at(point), point).distance_squared));
      }
      const auto middle = distances.begin() + distances.size() / 2;
      std::nth_element(distances.begin(), middle, distances.end());
      spacing = *middle;
      if (distances.size() % 2 == 0) {
        spacing = 0.5 * (spacing + *std::max_element(distances.begin(), middle));
      }
      spacing_squared_ = spacing * spacing;
      if (!std::isfinite(spacing_squared_) || spacing_squared_ <= 0) {
        throw std::invalid_argument("Point spacing must be positive and finite");
      }
    }

    std::size_t queryCount() const { return std::min(points_.size(), query_limit_); }
    std::size_t queryIndex(std::size_t i) const {
      return (i * points_.size() + points_.size() / 2) / queryCount();
    }
    Eigen::Map<const Eigen::Vector3d> at(std::size_t i) const {
      return Eigen::Map<const Eigen::Vector3d>(points_[i].data());
    }
    double spacingSquared() const { return spacing_squared_; }

    Nearest nearest(const Eigen::Vector3d& query, std::size_t skip = kNoPoint,
                    std::size_t seed = kNoPoint) const {
      Nearest best;
      if (query.allFinite()) {
        // A previous neighbor provides a valid upper bound at the new pose.
        if (seed != kNoPoint && seed != skip) {
          best = {(query - at(seed)).squaredNorm(), seed};
        }
        Eigen::Vector3d box_distance = Eigen::Vector3d::Zero();
        search(query, skip, 0, points_.size(), 0, box_distance, best);
      }
      return best;
    }

  private:
    bool containsQuery(std::size_t begin, std::size_t end) const {
      const auto size = points_.size();
      const auto count = queryCount();
      if (count == size) {
        return true;
      }
      // Invert queryIndex to find the first sampled rank at or after begin.
      const auto scaled_begin = begin * count;
      const auto first = scaled_begin <= size / 2
                             ? 0
                             : (scaled_begin - size / 2 + size - 1) / size;
      return first < count && queryIndex(first) < end;
    }

    // The midpoint of each interval is the KD-tree node, with its child ranges on either side.
    void build(std::size_t begin, std::size_t end, int axis) {
      // A leaf is scanned exhaustively. Only partition its points when the old
      // layout determines the position of a sampled query within that leaf.
      if (end - begin < 2 || (end - begin <= kLeafPoints && !containsQuery(begin, end))) {
        return;
      }
      const auto middle = begin + (end - begin) / 2;
      std::nth_element(points_.begin() + begin, points_.begin() + middle, points_.begin() + end,
                       [axis](const Point& a, const Point& b) { return a[axis] < b[axis]; });
      const auto next_axis = (axis + 1) % 3;
      build(begin, middle, next_axis);
      build(middle + 1, end, next_axis);
    }

    void search(const Eigen::Vector3d& query, std::size_t skip, std::size_t begin,
                std::size_t end, int axis, Eigen::Vector3d& box_distance, Nearest& best) const {
      if (end - begin <= kLeafPoints) {
        // Sampled query ranks match the original tree. Other leaf points can be unordered.
        for (std::size_t i = begin; i < end; ++i) {
          const double distance = (query - at(i)).squaredNorm();
          if (distance < best.distance_squared && i != skip) {
            best = {distance, i};
          }
        }
        return;
      }
      const auto middle = begin + (end - begin) / 2;
      const auto point = at(middle);
      if (middle != skip) {
        const double distance = (query - point).squaredNorm();
        if (distance < best.distance_squared) {
          best = {distance, middle};
        }
      }
      const double delta = query[axis] - point[axis];
      const auto next_axis = (axis + 1) % 3;
      const bool left = delta < 0;
      search(query, skip, left ? begin : middle + 1, left ? middle : end, next_axis, box_distance,
             best);
      const double delta_squared = delta * delta;
      if (delta_squared < best.distance_squared) {
        // Keep bounds from all ancestor splits to prune the far subtree exactly.
        const double previous = box_distance[axis];
        box_distance[axis] = std::max(previous, delta_squared);
        if (box_distance.sum() < best.distance_squared) {
          search(query, skip, left ? middle + 1 : begin, left ? end : middle, next_axis,
                 box_distance, best);
        }
        box_distance[axis] = previous;
      }
    }

    std::vector<Point> points_;
    std::size_t query_limit_;
    double spacing_squared_;
  };

  PoseScorer(IndexedSample source, IndexedSample target)
      : source_(std::move(source)), target_(std::move(target)),
        source_neighbors_(source_.queryCount(), kNoPoint),
        target_neighbors_(target_.queryCount(), kNoPoint) {}

  static std::vector<Point> voxelSample(const std::vector<Point>& points, double side) {
    std::size_t capacity = 1;
    while (capacity < 2 * points.size()) {
      capacity <<= 1;
    }
    std::vector<std::uint64_t> table(capacity, 0);
    const auto mask = capacity - 1;
    const double inverse_side = 1 / side;
    constexpr double bias = double(1 << 20);
    std::vector<Point> result;
    result.reserve(points.size());
    // Packed keys cover ordinary coordinates. Large cells use tuple keys.
    std::set<Point> wide_cells;
    for (const auto& point : points) {
      const Point cell{std::floor(point[0] * inverse_side), std::floor(point[1] * inverse_side),
                       std::floor(point[2] * inverse_side)};
      if (!std::all_of(cell.begin(), cell.end(), [](double value) {
            return value >= -bias && value < bias;
          })) {
        if (wide_cells.insert(cell).second) {
          result.push_back(point);
        }
        continue;
      }
      const auto x = static_cast<std::uint64_t>(cell[0] + bias);
      const auto y = static_cast<std::uint64_t>(cell[1] + bias);
      const auto z = static_cast<std::uint64_t>(cell[2] + bias);
      const std::uint64_t key = ((x << 42) | (y << 21) | z) + 1;
      auto hash = key;
      hash ^= hash >> 30;
      hash *= 0xbf58476d1ce4e5b9ULL;
      hash ^= hash >> 27;
      hash *= 0x94d049bb133111ebULL;
      hash ^= hash >> 31;
      auto bucket = static_cast<std::size_t>(hash) & mask;
      while (table[bucket] != 0 && table[bucket] != key) {
        bucket = (bucket + 1) & mask;
      }
      if (table[bucket] == 0) {
        table[bucket] = key;
        result.push_back(point);
      }
    }
    return result;
  }

  static IndexedSample verificationIndex(const Points& cloud) {
    const auto pilot = stratifiedSample(cloud, 8192);
    const double spacing = std::sqrt(IndexedSample(pilot, 1024, 128).spacingSquared());
    const auto points = stratifiedSample(cloud, 32768);
    const auto filtered = voxelSample(points, 2 * spacing);
    // A small cloud can fit in one voxel. Its original points still define a score.
    return IndexedSample(filtered.size() < 2 ? points : filtered, 2048, 512, spacing);
  }

  static double directedScore(const IndexedSample& source, const IndexedSample& target,
                              const Eigen::Matrix3d& linear, const Eigen::Vector3d& translation,
                              std::vector<std::size_t>& neighbors, double other_upper,
                              double incumbent) {
    const auto count = source.queryCount();
    const double spacing = target.spacingSquared();
    double total = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const Eigen::Vector3d point = linear * source.at(source.queryIndex(i)) + translation;
      const auto nearest = target.nearest(point, kNoPoint, neighbors[i]);
      neighbors[i] = nearest.point;
      total += spacing / (spacing + nearest.distance_squared);
      if ((i + 1) % 64 == 0 && i + 1 < count) {
        // Each unseen contribution is at most one. Rounding the partial sum up
        // to an integer bounds even the floating-point accumulation of the rest.
        const double upper = (std::ceil(total) + (count - i - 1)) / count;
        if (0.5 * (upper + other_upper) < incumbent) {
          return -std::numeric_limits<double>::infinity();
        }
      }
    }
    return total / count;
  }

  IndexedSample source_;
  IndexedSample target_;
  std::vector<std::size_t> source_neighbors_;
  std::vector<std::size_t> target_neighbors_;
  double previous_scale_ = std::numeric_limits<double>::quiet_NaN();
  Eigen::Matrix3d previous_rotation_;
  Eigen::Vector3d previous_translation_;
  double previous_score_;
};

} // namespace detail
} // namespace teaser
