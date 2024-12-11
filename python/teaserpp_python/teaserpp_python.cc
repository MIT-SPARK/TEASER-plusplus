/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include <omp.h>
#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "teaser/registration.h"
#include "teaser/certification.h"

namespace py = pybind11;

/**
 * Python interface with pybind11
 */
PYBIND11_MODULE(_teaserpp, m) {
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

  m.attr("OMP_MAX_THREADS") = omp_get_max_threads();

  // Python bound for teaser::RobustRegistrationSolver::ROTATION_ESTIMATE_ALGORITHM
  py::enum_<teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM>(
        m, "RotationEstimationAlgorithm"
      )
      .value("GNC_TLS", teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS)
      .value("FGR", teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR)
      .value("QUATRO", teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO);

  // Python bound for teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION
  py::enum_<teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION>(
        m, "InlierGraphFormulation"
      )
      .value("CHAIN", teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN)
      .value("COMPLETE", teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::COMPLETE);

  // Python bound for teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE
  py::enum_<teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE>(
        m, "InlierSelectionMode"
      )
      .value("PMC_EXACT", teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT)
      .value("PMC_HEU", teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU)
      .value("KCORE_HEU", teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::KCORE_HEU)
      .value("NONE", teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::NONE);

  // Python bound for DRSCertifier::EIG_SOLVER_TYPE
  py::enum_<teaser::DRSCertifier::EIG_SOLVER_TYPE>(m, "EigSolverType")
      .value("EIGEN", teaser::DRSCertifier::EIG_SOLVER_TYPE::EIGEN)
      .value("SPECTRA", teaser::DRSCertifier::EIG_SOLVER_TYPE::SPECTRA);

  // Python bound for teaser::RobustRegistraionSolver
  py::class_<teaser::RobustRegistrationSolver> solver(
      m, "RobustRegistrationSolver", py::dynamic_attr()
  );

  // Python bound for teaser::RobustRegistrationSolver functions
  solver.def(py::init<const teaser::RobustRegistrationSolver::Params&>())
        .def(py::init<double, double, bool,
                    teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM,
                    double, size_t, double,
                    teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION,
                    teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE,
                    double, bool, bool,
                    double, int>(),
           py::arg("noise_bound") = 0.01,
           py::arg("cbar2") = 1,
           py::arg("estimate_scaling") = true,
           py::arg("rotation_estimation_algorithm") =
             teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS,
           py::arg("rotation_gnc_factor") = 1.4,
           py::arg("rotation_max_iterations") = 100,
           py::arg("rotation_cost_threshold") = 1e-6,
           py::arg("rotation_tim_graph") =
             teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN,
           py::arg("inlier_selection_mode") =
             teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT,
           py::arg("kcore_heuristic_threshold") = 0.5,
           py::arg("use_max_clique") = true,
           py::arg("max_clique_exact_solution") = true,
           py::arg("max_clique_time_limit") = 3000,
           py::arg("max_clique_num_threads") = omp_get_max_threads())
      .def("getParams", &teaser::RobustRegistrationSolver::getParams)
      .def("solve", py::overload_cast<const Eigen::Matrix<double, 3, Eigen::Dynamic>&,
                                      const Eigen::Matrix<double, 3, Eigen::Dynamic>&>(
                        &teaser::RobustRegistrationSolver::solve))

      .def_property_readonly("solution", &teaser::RobustRegistrationSolver::getSolution)
      .def("getSolution", &teaser::RobustRegistrationSolver::getSolution)

      .def_property_readonly("gnc_rotation_cost_at_termination",
           &teaser::RobustRegistrationSolver::getGNCRotationCostAtTermination)
      .def("getGNCRotationCostAtTermination",
           &teaser::RobustRegistrationSolver::getGNCRotationCostAtTermination)

      .def_property_readonly("scale_inliers_mask", &teaser::RobustRegistrationSolver::getScaleInliersMask)
      .def("getScaleInliersMask", &teaser::RobustRegistrationSolver::getScaleInliersMask)

      .def_property_readonly("scale_inliers_map", &teaser::RobustRegistrationSolver::getScaleInliersMap)
      .def("getScaleInliersMap", &teaser::RobustRegistrationSolver::getScaleInliersMap)

      .def_property_readonly("scale_inliers", &teaser::RobustRegistrationSolver::getScaleInliers)
      .def("getScaleInliers", &teaser::RobustRegistrationSolver::getScaleInliers)

      .def_property_readonly("rotation_inliers_mask", &teaser::RobustRegistrationSolver::getRotationInliersMask)
      .def("getRotationInliersMask", &teaser::RobustRegistrationSolver::getRotationInliersMask)

      .def("getRotationInliersMap", &teaser::RobustRegistrationSolver::getRotationInliersMap)

      .def_property_readonly("rotation_inliers", &teaser::RobustRegistrationSolver::getRotationInliers)
      .def("getRotationInliers", &teaser::RobustRegistrationSolver::getRotationInliers)

      .def_property_readonly("translation_inliers_mask",
           &teaser::RobustRegistrationSolver::getTranslationInliersMask)
      .def("getTranslationInliersMask",
           &teaser::RobustRegistrationSolver::getTranslationInliersMask)


      .def_property_readonly("translation_inliers_map", &teaser::RobustRegistrationSolver::getTranslationInliersMap)
      .def("getTranslationInliersMap", &teaser::RobustRegistrationSolver::getTranslationInliersMap)

      .def_property_readonly("translation_inliers", &teaser::RobustRegistrationSolver::getTranslationInliers)
      .def("getTranslationInliers", &teaser::RobustRegistrationSolver::getTranslationInliers)

      .def_property_readonly("inlier_max_clique", &teaser::RobustRegistrationSolver::getInlierMaxClique)
      .def("getInlierMaxClique", &teaser::RobustRegistrationSolver::getInlierMaxClique)

      .def_property_readonly("inlier_graph", &teaser::RobustRegistrationSolver::getInlierGraph)
      .def("getInlierGraph", &teaser::RobustRegistrationSolver::getInlierGraph)

      .def_property_readonly("src_tims_map", &teaser::RobustRegistrationSolver::getSrcTIMsMap)
      .def("getSrcTIMsMap", &teaser::RobustRegistrationSolver::getSrcTIMsMap)

      .def_property_readonly("dst_tims_map", &teaser::RobustRegistrationSolver::getDstTIMsMap)
      .def("getDstTIMsMap", &teaser::RobustRegistrationSolver::getDstTIMsMap)

      .def_property_readonly("src_tims_map_for_rotation", &teaser::RobustRegistrationSolver::getSrcTIMsMapForRotation)
      .def("getSrcTIMsMapForRotation", &teaser::RobustRegistrationSolver::getSrcTIMsMapForRotation)

      .def_property_readonly("dst_tims_map_for_rotation", &teaser::RobustRegistrationSolver::getSrcTIMsMapForRotation)
      .def("getDstTIMsMapForRotation", &teaser::RobustRegistrationSolver::getDstTIMsMapForRotation)

      .def_property_readonly("max_clique_src_tims", &teaser::RobustRegistrationSolver::getMaxCliqueSrcTIMs)
      .def("getMaxCliqueSrcTIMs", &teaser::RobustRegistrationSolver::getMaxCliqueSrcTIMs)

      .def_property_readonly("max_clique_dst_tims", &teaser::RobustRegistrationSolver::getMaxCliqueSrcTIMs)
      .def("getMaxCliqueDstTIMs", &teaser::RobustRegistrationSolver::getMaxCliqueDstTIMs)

      .def_property_readonly("src_tims", &teaser::RobustRegistrationSolver::getSrcTIMs)
      .def("getSrcTIMs", &teaser::RobustRegistrationSolver::getSrcTIMs)

      .def_property_readonly("dst_tims", &teaser::RobustRegistrationSolver::getSrcTIMs)
      .def("getDstTIMs", &teaser::RobustRegistrationSolver::getDstTIMs);

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

        std::string rot_alg;
        if (a.rotation_estimation_algorithm ==
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR) {
          rot_alg = "FGR";
        }
        if (a.rotation_estimation_algorithm ==
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS) {
          rot_alg = "GNC_TLS";
        }
        if (a.rotation_estimation_algorithm ==
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO) {
          rot_alg = "QUATRO";
        }

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
