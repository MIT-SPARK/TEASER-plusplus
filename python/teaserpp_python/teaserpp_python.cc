/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <string>
#include <sstream>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "teaser/registration.h"
#include "teaser/certification.h"

namespace py = pybind11;

/**
 * Convenience function for solving a batch of registration problems with TEASER++.
 * Suitable for PyTorch users who want to use TEASER++. This function does not use maximum clique.
 */
std::vector<std::pair<teaser::RegistrationSolution, Eigen::Matrix<bool, 1, Eigen::Dynamic>>>
batch_gnc_solve(const std::vector<py::EigenDRef<Eigen::Matrix<double, 3, Eigen::Dynamic>>>& src,
                const std::vector<py::EigenDRef<Eigen::Matrix<double, 3, Eigen::Dynamic>>>& dst,
                const std::vector<double>& noise_bound, bool estimate_scale) {

  size_t B = src.size();
  assert(B == noise_bound.size());

  std::vector<std::pair<teaser::RegistrationSolution, Eigen::Matrix<bool, 1, Eigen::Dynamic>>> sols;
  sols.resize(B);
#pragma omp parallel for default(none) shared(B, noise_bound, src, dst, sols, std::cout)           \
    firstprivate(estimate_scale)
  for (size_t i = 0; i < B; ++i) {
    teaser::RobustRegistrationSolver::Params params;
    params.cbar2 = 1;
    params.estimate_scaling = estimate_scale;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
    params.rotation_cost_threshold = 0.005;
    params.max_clique_num_threads = 15;
    params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE;
    params.noise_bound = noise_bound[i];

    // Prepare the solver object
    teaser::RobustRegistrationSolver solver(params);

    // Solve
    sols[i].first = solver.solve(src[i], dst[i]);
    sols[i].second = solver.getTranslationInliersMask();
  }

  return sols;
}

/**
 * Python interface with pybind11
 */
PYBIND11_MODULE(teaserpp_python, m) {
  m.doc() = "Python binding for TEASER++";

  // Python bound for teaser::RegistrationSolution
  py::class_<teaser::RegistrationSolution>(m, "RegistrationSolution")
      .def_readwrite("scale", &teaser::RegistrationSolution::scale)
      .def_readwrite("translation", &teaser::RegistrationSolution::translation)
      .def_readwrite("rotation", &teaser::RegistrationSolution::rotation)
      .def("__repr__", [](const teaser::RegistrationSolution& a) {
        std::ostringstream print_string;

        print_string << "<RegistrationSolution with scale=" << a.scale << "\n"
                     << "translation=\n"
                     << a.translation << "\n"
                     << "rotation=\n"
                     << a.rotation << "\n"
                     << ">";
        return print_string.str();
      });

  // Python bound for teaser::RobustRegistraionSolver
  py::class_<teaser::RobustRegistrationSolver> solver(m, "RobustRegistrationSolver");

  // Python bound for teaser::RobustRegistrationSolver functions
  solver.def(py::init<>())
      .def(py::init<const teaser::RobustRegistrationSolver::Params&>())
      .def("getParams", &teaser::RobustRegistrationSolver::getParams)
      .def("reset", &teaser::RobustRegistrationSolver::reset)
      .def("solve", py::overload_cast<const Eigen::Matrix<double, 3, Eigen::Dynamic>&,
                                      const Eigen::Matrix<double, 3, Eigen::Dynamic>&>(
                        &teaser::RobustRegistrationSolver::solve))
      .def("getSolution", &teaser::RobustRegistrationSolver::getSolution)
      .def("getGNCRotationCostAtTermination",
           &teaser::RobustRegistrationSolver::getGNCRotationCostAtTermination)
      .def("getScaleInliersMask", &teaser::RobustRegistrationSolver::getScaleInliersMask)
      .def("getScaleInliersMap", &teaser::RobustRegistrationSolver::getScaleInliersMap)
      .def("getScaleInliers", &teaser::RobustRegistrationSolver::getScaleInliers)
      .def("getRotationInliersMask", &teaser::RobustRegistrationSolver::getRotationInliersMask)
      .def("getRotationInliersMap", &teaser::RobustRegistrationSolver::getRotationInliersMap)
      .def("getRotationInliers", &teaser::RobustRegistrationSolver::getRotationInliers)
      .def("getTranslationInliersMask",
           &teaser::RobustRegistrationSolver::getTranslationInliersMask)
      .def("getTranslationInliersMap", &teaser::RobustRegistrationSolver::getTranslationInliersMap)
      .def("getTranslationInliers", &teaser::RobustRegistrationSolver::getTranslationInliers)
      .def("getInlierMaxClique", &teaser::RobustRegistrationSolver::getInlierMaxClique)
      .def("getInlierGraph", &teaser::RobustRegistrationSolver::getInlierGraph)
      .def("getSrcTIMsMap", &teaser::RobustRegistrationSolver::getSrcTIMsMap)
      .def("getDstTIMsMap", &teaser::RobustRegistrationSolver::getDstTIMsMap)
      .def("getSrcTIMsMapForRotation", &teaser::RobustRegistrationSolver::getSrcTIMsMapForRotation)
      .def("getDstTIMsMapForRotation", &teaser::RobustRegistrationSolver::getDstTIMsMapForRotation)
      .def("getMaxCliqueSrcTIMs", &teaser::RobustRegistrationSolver::getMaxCliqueSrcTIMs)
      .def("getMaxCliqueDstTIMs", &teaser::RobustRegistrationSolver::getMaxCliqueDstTIMs)
      .def("getSrcTIMs", &teaser::RobustRegistrationSolver::getSrcTIMs)
      .def("getDstTIMs", &teaser::RobustRegistrationSolver::getDstTIMs);

  // Python bound for teaser::RobustRegistrationSolver::ROTATION_ESTIMATE_ALGORITHM
  py::enum_<teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM>(
      solver, "ROTATION_ESTIMATION_ALGORITHM")
      .value("GNC_TLS", teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS)
      .value("FGR", teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR);

  // Python bound for teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION
  py::enum_<teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION>(solver,
                                                                        "INLIER_GRAPH_FORMULATION")
      .value("CHAIN", teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN)
      .value("COMPLETE", teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::COMPLETE);

  // Python bound for teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE
  py::enum_<teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE>(solver,
                                                                     "INLIER_SELECTION_MODE")
      .value("PMC_EXACT", teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT)
      .value("PMC_HEU", teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU)
      .value("KCORE_HEU", teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::KCORE_HEU)
      .value("NONE", teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE);

  // Python bound for teaser::RobustRegistrationSolver::Params
  py::class_<teaser::RobustRegistrationSolver::Params>(solver, "Params")
      .def(py::init<>())
      .def_readwrite("noise_bound", &teaser::RobustRegistrationSolver::Params::noise_bound)
      .def_readwrite("cbar2", &teaser::RobustRegistrationSolver::Params::cbar2)
      .def_readwrite("estimate_scaling",
                     &teaser::RobustRegistrationSolver::Params::estimate_scaling)
      .def_readwrite("rotation_estimation_algorithm",
                     &teaser::RobustRegistrationSolver::Params::rotation_estimation_algorithm)
      .def_readwrite("rotation_gnc_factor",
                     &teaser::RobustRegistrationSolver::Params::rotation_gnc_factor)
      .def_readwrite("rotation_max_iterations",
                     &teaser::RobustRegistrationSolver::Params::rotation_max_iterations)
      .def_readwrite("rotation_tim_graph",
                     &teaser::RobustRegistrationSolver::Params::rotation_tim_graph)
      .def_readwrite("inlier_selection_mode",
                     &teaser::RobustRegistrationSolver::Params::inlier_selection_mode)
      .def_readwrite("kcore_heuristic_threshold",
                     &teaser::RobustRegistrationSolver::Params::kcore_heuristic_threshold)
      .def_readwrite("rotation_cost_threshold",
                     &teaser::RobustRegistrationSolver::Params::rotation_cost_threshold)
      .def_readwrite("use_max_clique", &teaser::RobustRegistrationSolver::Params::use_max_clique)
      .def_readwrite("max_clique_exact_solution",
                     &teaser::RobustRegistrationSolver::Params::max_clique_exact_solution)
      .def_readwrite("max_clique_time_limit",
                     &teaser::RobustRegistrationSolver::Params::max_clique_time_limit)
      .def("__repr__", [](const teaser::RobustRegistrationSolver::Params& a) {
        std::ostringstream print_string;

        std::string rot_alg =
            a.rotation_estimation_algorithm ==
                    teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR
                ? "FGR"
                : "GNC-TLS";

        std::string inlier_selection_alg;
        if (a.inlier_selection_mode ==
            teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT) {
          inlier_selection_alg = "PMC_EXACT";
        }
        if (a.inlier_selection_mode ==
            teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU) {
          inlier_selection_alg = "PMC_HEU";
        }
        if (a.inlier_selection_mode ==
            teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::KCORE_HEU) {
          inlier_selection_alg = "KCORE_HEU";
        }
        if (a.inlier_selection_mode ==
            teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE) {
          inlier_selection_alg = "NONE";
        }

        print_string << "<Params with noise_bound=" << a.noise_bound << "\n"
                     << "cbar2=" << a.cbar2 << "\n"
                     << "estimate_sccaling=" << a.estimate_scaling << "\n"
                     << "rotation_estimation_algorithm=" << rot_alg << "\n"
                     << "inlier_selection_mode=" << inlier_selection_alg << "\n"
                     << ">";
        return print_string.str();
      });

  // Convenience function for solving batched problems with TEASER++ in parallel
  m.def("batch_gnc_solve", &batch_gnc_solve, "Solve a batch of problems in parallel",
        py::arg("src"), py::arg("dst"), py::arg("noise_bound"), py::arg("estimate_scale"));

  // Python bound for CertificationResult
  py::class_<teaser::CertificationResult>(m, "CertificationResult")
      .def_readwrite("is_optimal", &teaser::CertificationResult::is_optimal)
      .def_readwrite("best_suboptimality", &teaser::CertificationResult::best_suboptimality)
      .def_readwrite("suboptimality_traj", &teaser::CertificationResult::suboptimality_traj)
      .def("__repr__", [](const teaser::CertificationResult& a) {
        std::ostringstream print_string;

        print_string << "<CertificationResult \n"
                     << "Is optimal:" << a.is_optimal << "\n"
                     << "Best suboptimality:" << a.best_suboptimality << "\n"
                     << "Iterations: " << a.suboptimality_traj.size() << "\n"
                     << ">";
        return print_string.str();
      });

  // Python bound for DRSCertifier
  py::class_<teaser::DRSCertifier> certifier(m, "DRSCertifier");
  certifier.def(py::init<const teaser::DRSCertifier::Params>())
      .def(
          "certify",
          py::overload_cast<const Eigen::Matrix3d&, const Eigen::Matrix<double, 3, Eigen::Dynamic>&,
                            const Eigen::Matrix<double, 3, Eigen::Dynamic>&,
                            const Eigen::Matrix<bool, 1, Eigen::Dynamic>&>(
              &teaser::DRSCertifier::certify))
      .def(
          "certify",
          py::overload_cast<const Eigen::Matrix3d&, const Eigen::Matrix<double, 3, Eigen::Dynamic>&,
                            const Eigen::Matrix<double, 3, Eigen::Dynamic>&,
                            const Eigen::Matrix<double, 1, Eigen::Dynamic>&>(
              &teaser::DRSCertifier::certify));

  // Python bound for DRSCertifier::EIG_SOLVER_TYPE
  py::enum_<teaser::DRSCertifier::EIG_SOLVER_TYPE>(certifier, "EIG_SOLVER_TYPE")
      .value("EIGEN", teaser::DRSCertifier::EIG_SOLVER_TYPE::EIGEN)
      .value("SPECTRA", teaser::DRSCertifier::EIG_SOLVER_TYPE::SPECTRA);

  // Python bound for DRSCertifier parameter struct
  py::class_<teaser::DRSCertifier::Params>(certifier, "Params")
      .def(py::init<>())
      .def_readwrite("noise_bound", &teaser::DRSCertifier::Params::noise_bound)
      .def_readwrite("cbar2", &teaser::DRSCertifier::Params::cbar2)
      .def_readwrite("sub_optimality", &teaser::DRSCertifier::Params::sub_optimality)
      .def_readwrite("max_iterations", &teaser::DRSCertifier::Params::max_iterations)
      .def_readwrite("gamma_tau", &teaser::DRSCertifier::Params::gamma_tau)
      .def_readwrite("eig_decomposition_solver",
                     &teaser::DRSCertifier::Params::eig_decomposition_solver);
}