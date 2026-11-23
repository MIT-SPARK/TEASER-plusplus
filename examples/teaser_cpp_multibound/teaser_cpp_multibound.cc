#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <teaser/ply_io.h>
#include <teaser/registration.h>

namespace {

using Solver = teaser::RobustRegistrationSolver;
using Matches = std::vector<std::pair<int, int>>;

teaser::PointCloud readCloud(const char* path) {
  teaser::PointCloud cloud;
  if (teaser::PLYReader().read(path, cloud) != 0 || cloud.size() < 3) {
    throw std::runtime_error(std::string("Expected a PLY cloud with at least three points: ") +
                             path);
  }
  for (const auto& p : cloud) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      throw std::runtime_error(std::string("Non-finite point in ") + path);
    }
  }
  return cloud;
}

Matches readMatches(const char* path, const teaser::PointCloud& source,
                                     const teaser::PointCloud& target) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error(std::string("Cannot open correspondence file: ") + path);
  }
  Matches indices;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream row(line);
    if ((row >> std::ws).eof()) {
      continue;
    }
    int src, dst;
    if (!(row >> src >> dst) || !(row >> std::ws).eof() ||
        src < 0 || dst < 0 || static_cast<std::size_t>(src) >= source.size() ||
        static_cast<std::size_t>(dst) >= target.size()) {
      throw std::runtime_error("Expected pairs of in-range, zero-based point indices");
    }
    indices.emplace_back(src, dst);
  }
  if (!file.eof() || indices.size() < 3) {
    throw std::runtime_error("Expected at least three pairs of point indices");
  }

  return indices;
}

using Points = Eigen::Matrix<double, 3, Eigen::Dynamic>;

Points pointMatrix(const teaser::PointCloud& cloud) {
  Points points(3, cloud.size());
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const auto& p = cloud[i];
    points.col(i) << p.x, p.y, p.z;
  }
  return points;
}

double positiveNumber(const char* text) {
  std::size_t end = 0;
  const double value = std::stod(text, &end);
  if (text[end] != '\0' || !std::isfinite(value) || value <= 0) {
    throw std::invalid_argument(std::string("Expected a positive, finite value: ") + text);
  }
  return value;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " source.ply target.ply matches.txt (auto | noise_bound [noise_bound ...])"
              << " [--verify source_verify.ply target_verify.ply]\n";
    return 2;
  }
  try {
    int bounds_end = 4;
    while (bounds_end < argc && std::string(argv[bounds_end]) != "--verify") {
      ++bounds_end;
    }
    const bool verify = bounds_end < argc;
    if (bounds_end == 4 || (verify && argc - bounds_end != 3)) {
      throw std::invalid_argument("Expected a bound and two PLY paths after --verify");
    }
    const bool automatic = bounds_end == 5 && std::string(argv[4]) == "auto";
    std::vector<double> bounds;
    if (!automatic) {
      for (int i = 4; i < bounds_end; ++i) {
        bounds.push_back(positiveNumber(argv[i]));
      }
    }
    const auto source = readCloud(argv[1]);
    const auto target = readCloud(argv[2]);
    const auto matches = readMatches(argv[3], source, target);

    Solver::Params params;
    params.estimate_scaling = false;
    Solver solver(params);
    const auto best = [&] {
      if (verify && (automatic || bounds.size() > 1)) {
        Points src(3, matches.size()), dst(3, matches.size());
        for (std::size_t i = 0; i < matches.size(); ++i) {
          const auto& p = source[matches[i].first];
          const auto& q = target[matches[i].second];
          src.col(i) << p.x, p.y, p.z;
          dst.col(i) << q.x, q.y, q.z;
        }
        const auto verification_src = pointMatrix(readCloud(argv[bounds_end + 1]));
        const auto verification_dst = pointMatrix(readCloud(argv[bounds_end + 2]));
        return automatic
                   ? solver.solve(src, dst, Solver::NoiseBoundMode::AUTO, verification_src,
                                  verification_dst)
                   : solver.solve(src, dst, bounds, verification_src, verification_dst);
      }
      return automatic ? solver.solve(source, target, matches, Solver::NoiseBoundMode::AUTO)
                       : solver.solve(source, target, matches, bounds);
    }();
    if (!best.valid) {
      throw std::runtime_error("No noise bound produced a valid rigid transform");
    }

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.topLeftCorner<3, 3>() = best.rotation;
    transform.topRightCorner<3, 1>() = best.translation;
    std::cout << "noise_bound " << solver.getParams().noise_bound << '\n'
              << "source_to_target\n"
              << std::setprecision(std::numeric_limits<double>::max_digits10) << transform << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
