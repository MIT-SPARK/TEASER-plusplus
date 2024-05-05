// An example showing TEASER++ registration with FPFH features with the Stanford bunny model
#include <chrono>
#include <iostream>
#include <random>

#include <Eigen/Core>

#include <teaser/ply_io.h>
#include <teaser/registration.h>
#include <teaser/matcher.h>

// Macro constants for generating noise and outliers
#define NOISE_BOUND 0.05

inline double getAngularError(Eigen::Matrix3d R_exp, Eigen::Matrix3d R_est) {
  return std::abs(std::acos(fmin(fmax(((R_exp.transpose() * R_est).trace() - 1) / 2, -1.0), 1.0)));
}

inline void calcErrors(const Eigen::Matrix4d& T, const Eigen::Matrix3d est_rot,
                       const Eigen::Vector3d est_ts, double& rot_error, double& ts_error) {
  rot_error = getAngularError(T.topLeftCorner(3, 3), est_rot);
  ts_error = (T.topRightCorner(3, 1) - est_ts).norm();
}

inline void getParams(const double noise_bound, const std::string reg_type,
                      teaser::RobustRegistrationSolver::Params& params) {
  params.noise_bound = noise_bound;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  if (reg_type == "Quatro") {
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO;
    params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;
  } else if (reg_type == "TEASER") {
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
  } else {
    throw std::invalid_argument("Not implemented!");
  }
  params.rotation_cost_threshold = 0.0002;
}

inline Eigen::Matrix3d get3DRot(const double yaw_deg, const double pitch_deg, const double roll_deg) {
  double yaw = yaw_deg * M_PI / 180.0;
  double pitch = pitch_deg * M_PI / 180.0;
  double roll = roll_deg * M_PI / 180.0;

  Eigen::Matrix3d yaw_mat = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d pitch_mat = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d roll_mat = Eigen::Matrix3d::Identity();

  double cy = cos(yaw);
  double sy = sin(yaw);
  yaw_mat(0, 0) = cy; yaw_mat(0, 1) = -sy;
  yaw_mat(1, 0) = sy; yaw_mat(1, 1) = cy;
  double cp = cos(pitch);
  double sp = sin(pitch);
  pitch_mat(0, 0) = cp; pitch_mat(0, 2) = sp;
  pitch_mat(2, 0) = -sp; pitch_mat(2, 2) = cp;
  double cr = cos(roll);
  double sr = sin(roll);
  roll_mat(1, 1) = cr; roll_mat(1, 2) = -sr;
  roll_mat(2, 1) = sr; roll_mat(2, 2) = cr;

  // 3D Rotation: R_z * R_y * R_x
  return yaw_mat * pitch_mat * roll_mat;
}

int main() {
  std::cout << "\033[1;33m================================== Example of Quatro =================================" << std::endl;
  std::cout << "NOTE that this example does not mean that Quatro is better than TEASER++!" << std::endl;
  std::cout << "Quatro is a specialized version of TEASER++, which forgoes roll and pitch estimation." << std::endl;
  std::cout << "Instead, Quatro shows promising performance in the case where the yaw angle is dominant, " << std::endl;
  std::cout << "so it is suitable for autonomous vehicles or terrestrial mobile robots." << std::endl;
  std::cout << "======================================================================================\033[0m" << std::endl;

  //   Load the .ply file
  teaser::PLYReader reader;
  teaser::PointCloud src_cloud;
  auto status = reader.read("./example_data/bun_zipper_res3.ply", src_cloud);
  int N = src_cloud.size();

  // Convert the point cloud to Eigen
  Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, N);
  for (size_t i = 0; i < N; ++i) {
    src.col(i) << src_cloud[i].x, src_cloud[i].y, src_cloud[i].z;
  }

  // Homogeneous coordinates
  Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
  src_h.resize(4, src.cols());
  src_h.topRows(3) = src;
  src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

  // Apply an arbitrary SE(3) transformation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist_for_yaw(179, 179);
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  Eigen::Matrix3d random_rot = get3DRot(static_cast<double>(dist_for_yaw(gen)), 0, 0); // yaw, pitch, and roll order
  T.block<3, 3>(0, 0) = random_rot;
  T(0, 3) = -1.15576939e-01;
  T(1, 3) = -3.87705398e-02;
  T(2, 3) =  1.14874890e-01;

  // Apply transformation
  Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt = tgt_h.topRows(3);

  // Convert to teaser point cloud
  teaser::PointCloud tgt_cloud;
  for (size_t i = 0; i < tgt.cols(); ++i) {
    tgt_cloud.push_back({static_cast<float>(tgt(0, i)), static_cast<float>(tgt(1, i)),
                         static_cast<float>(tgt(2, i))});
  }

  // Compute FPFH
  teaser::FPFHEstimation fpfh;
  auto obj_descriptors = fpfh.computeFPFHFeatures(src_cloud, 0.02, 0.04);
  auto scene_descriptors = fpfh.computeFPFHFeatures(tgt_cloud, 0.02, 0.04);

  teaser::Matcher matcher;
  auto correspondences = matcher.calculateCorrespondences(
      src_cloud, tgt_cloud, *obj_descriptors, *scene_descriptors, false, true, false, 0.95);

  // Prepare solver parameters
  teaser::RobustRegistrationSolver::Params quatro_param, teaser_param;
  getParams(NOISE_BOUND / 2, "Quatro", quatro_param);
  std::chrono::steady_clock::time_point begin_q = std::chrono::steady_clock::now();
  teaser::RobustRegistrationSolver Quatro(quatro_param);
  Quatro.solve(src_cloud, tgt_cloud, correspondences);
  std::chrono::steady_clock::time_point end_q = std::chrono::steady_clock::now();
  auto solution_by_quatro = Quatro.getSolution();

  std::cout << "=====================================" << std::endl;
  std::cout << "           Quatro Results            " << std::endl;
  std::cout << "=====================================" << std::endl;
  double rot_error_quatro, ts_error_quatro;
  calcErrors(T, solution_by_quatro.rotation, solution_by_quatro.translation,
             rot_error_quatro, ts_error_quatro);
  // Compare results
  std::cout << "Expected rotation: " << std::endl;
  std::cout << T.topLeftCorner(3, 3) << std::endl;
  std::cout << "Estimated rotation: " << std::endl;
  std::cout << solution_by_quatro.rotation << std::endl;
  std::cout << "Error (rad): " << rot_error_quatro << std::endl;
  std::cout << "Expected translation: " << std::endl;
  std::cout << T.topRightCorner(3, 1) << std::endl;
  std::cout << "Estimated translation: " << std::endl;
  std::cout << solution_by_quatro.translation << std::endl;
  std::cout << "Error (m): " << ts_error_quatro << std::endl;
  std::cout << "Time taken (s): "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_q - begin_q).count() /
                   1000000.0 << std::endl;
  std::cout << "=====================================" << std::endl;

  getParams(NOISE_BOUND / 2, "TEASER", teaser_param);
  std::chrono::steady_clock::time_point begin_t = std::chrono::steady_clock::now();
  teaser::RobustRegistrationSolver TEASER(teaser_param);
  TEASER.solve(src_cloud, tgt_cloud, correspondences);
  std::chrono::steady_clock::time_point end_t = std::chrono::steady_clock::now();
  auto solution_by_teaser = TEASER.getSolution();

  std::cout << "=====================================" << std::endl;
  std::cout << "          TEASER++ Results           " << std::endl;
  std::cout << "=====================================" << std::endl;
  double rot_error_teaser, ts_error_teaser;
  calcErrors(T, solution_by_teaser.rotation, solution_by_teaser.translation,
             rot_error_teaser, ts_error_teaser);
  // Compare results
  std::cout << "Expected rotation: " << std::endl;
  std::cout << T.topLeftCorner(3, 3) << std::endl;
  std::cout << "Estimated rotation: " << std::endl;
  std::cout << solution_by_teaser.rotation << std::endl;
  std::cout << "Error (rad): " << rot_error_teaser << std::endl;
  std::cout << "Expected translation: " << std::endl;
  std::cout << T.topRightCorner(3, 1) << std::endl;
  std::cout << "Estimated translation: " << std::endl;
  std::cout << solution_by_teaser.translation << std::endl;
  std::cout << "Error (m): " << ts_error_teaser << std::endl;
  std::cout << "Time taken (s): "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_t - begin_t).count() /
                   1000000.0 << std::endl;
  std::cout << "=====================================" << std::endl;
}
