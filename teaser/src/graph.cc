/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "teaser/graph.h"
#include "pmc_heuristic.h"
#include "pmc/pmc.h"

#include <algorithm>
#include <utility>

namespace {

int coloringUpperBound(const teaser::Graph& graph, const std::vector<int>& cores,
                       const std::vector<int>& core_order, int lower_bound, int color_limit,
                       const std::vector<int>& previous_clique) {
  // PMC stores core numbers plus one. Color only the subgraph that could
  // contain a larger clique, starting from the feasible previous clique.
  // Return zero if the coloring cannot improve the caller's stopping bound.
  std::vector<int> colors(graph.numVertices(), -1);
  int color_count = 0;
  for (int vertex : previous_clique) {
    if (cores[vertex] > lower_bound) {
      colors[vertex] = color_count++;
    }
  }
  if (color_count >= color_limit) {
    return 0;
  }

  std::vector<int> forbidden(color_limit, -1);
  for (int i = graph.numVertices() - 1; i >= 0; --i) {
    const int vertex = core_order[i];
    if (cores[vertex] <= lower_bound) {
      break;
    }
    if (colors[vertex] >= 0) {
      continue;
    }
    for (int neighbor : graph.getEdges(vertex)) {
      if (colors[neighbor] >= 0) {
        forbidden[colors[neighbor]] = vertex;
      }
    }
    const auto free_color = std::find_if(forbidden.begin(), forbidden.begin() + color_count,
                                         [vertex](int used) { return used != vertex; });
    const int color = static_cast<int>(free_color - forbidden.begin());
    if (color == color_count && ++color_count == color_limit) {
      return 0;
    }
    colors[vertex] = color;
  }
  return std::max(lower_bound, color_count);
}

std::vector<int> solvePmc(pmc::pmc_graph& G, const teaser::MaxCliqueSolver::Params& params,
                          const teaser::Graph* multi_bound_graph,
                          const std::vector<int>* previous_clique,
                          teaser::detail::PmcHeuristic* heuristic) {
  using CLIQUE_SOLVER_MODE = teaser::MaxCliqueSolver::CLIQUE_SOLVER_MODE;
  const int vertex_count = G.num_vertices();
  // Prepare PMC input
  // TODO: Incorporate this to the constructor
  pmc::input in;
  in.algorithm = 0;
  in.threads = params.num_threads;
  in.experiment = 0;
  in.lb = 0;
  in.ub = 0;
  in.param_ub = 0;
  in.adj_limit = 20000;
  in.time_limit = params.time_limit;
  in.remove_time = 4;
  in.graph_stats = false;
  in.verbose = false;
  in.help = false;
  in.MCE = false;
  in.decreasing_order = false;
  in.heu_strat = "kcore";
  in.vertex_search_order = "deg";

  // vector to represent max clique
  std::vector<int> C;

  // upper-bound of max clique
  G.compute_cores();
  auto max_core = G.get_max_core();

  TEASER_DEBUG_INFO_MSG("Max core number: " << max_core);
  TEASER_DEBUG_INFO_MSG("Num vertices: " << G.get_vertices().size());

  // check for k-core heuristic threshold
  // check whether threshold equals 1 to short circuit the comparison
  if (params.solver_mode == CLIQUE_SOLVER_MODE::KCORE_HEU &&
      params.kcore_heuristic_threshold != 1 &&
      max_core > static_cast<int>(params.kcore_heuristic_threshold *
                                  static_cast<double>(vertex_count))) {
    TEASER_DEBUG_INFO_MSG("Using K-core heuristic finder.");
    // remove all nodes with core number less than max core number
    // k_cores is a vector saving the core number of each vertex
    auto k_cores = G.get_kcores();
    for (int i = 1; i < k_cores->size(); ++i) {
      // Note: k_core has size equals to num vertices + 1
      if ((*k_cores)[i] >= max_core) {
        C.push_back(i - 1);
      }
    }
    return C;
  }

  if (in.ub == 0) {
    in.ub = max_core + 1;
  }

  // A previous clique is a valid lower bound on a nested graph. Use it directly
  // only if the core bound proves it is the unique maximum clique.
  const bool reuse_clique = multi_bound_graph && previous_clique &&
                           params.solver_mode == CLIQUE_SOLVER_MODE::PMC_EXACT;
  const auto& cores = *G.get_kcores();
  if (reuse_clique && previous_clique->size() == static_cast<std::size_t>(in.ub) &&
      std::count(cores.begin(), cores.begin() + vertex_count, in.ub) == in.ub) {
    return *previous_clique;
  }

  // lower-bound of max clique
  if (in.lb == 0 && in.heu_strat != "0") { // skip if given as input
    pmc::pmc_heu maxclique(G, in);
    const auto words = (static_cast<std::size_t>(vertex_count) + 63) / 64;
    // Keep the bitset no larger than the existing CSR edge storage.
    if (heuristic && params.num_threads == 1 && vertex_count < in.adj_limit &&
        2 * vertex_count * words <= G.get_edges().size()) {
      heuristic->prepare(G);
      in.lb = heuristic->search(G, maxclique, C);
    } else {
      in.lb = maxclique.search(G, C);
    }
  }

  assert(in.lb != 0);
  if (in.lb == 0) {
    // This means that max clique has a size of one
    TEASER_DEBUG_ERROR_MSG("Max clique lower bound equals to zero. Abort.");
    return C;
  }

  if (in.lb == in.ub) {
    return C;
  }

  // Keep the existing heuristic incumbent and ordering. With one thread a
  // tighter stopping bound returns the same first maximum found by PMC.
  const bool tighten_stop = params.num_threads == 1;
  if (reuse_clique &&
      (tighten_stop || previous_clique->size() <= static_cast<std::size_t>(in.lb))) {
    const int color_limit = tighten_stop ? in.ub : in.lb + 1;
    const int bound = coloringUpperBound(*multi_bound_graph, cores, *G.get_kcore_ordering(), in.lb,
                                         color_limit, *previous_clique);
    if (bound == in.lb) {
      return C;
    }
    if (tighten_stop && bound > 0) {
      in.param_ub = bound;
    }
  }

  // Optional exact max clique finding
  if (params.solver_mode == CLIQUE_SOLVER_MODE::PMC_EXACT) {
    // The following methods are used:
    // 1. k-core pruning
    // 2. neigh-core pruning/ordering
    // 3. dynamic coloring bounds/sort
    // see the original PMC paper and implementation for details:
    // R. A. Rossi, D. F. Gleich, and A. H. Gebremedhin, “Parallel Maximum Clique Algorithms with
    // Applications to Network Analysis,” SIAM J. Sci. Comput., vol. 37, no. 5, pp. C589–C616, Jan.
    // 2015.
    if (G.num_vertices() < in.adj_limit) {
      G.create_adj();
      pmc::pmcx_maxclique finder(G, in);
      finder.search_dense(G, C);
    } else {
      pmc::pmcx_maxclique finder(G, in);
      finder.search(G, C);
    }
  }

  return C;
}

} // namespace

std::vector<int> teaser::MaxCliqueSolver::findMaxClique(teaser::Graph graph) {

  // Handle deprecated field
  if (!params_.solve_exactly) {
    params_.solver_mode = CLIQUE_SOLVER_MODE::PMC_HEU;
  }

  // Create a PMC graph from the TEASER graph
  std::vector<int> edges;
  std::vector<long long> vertices;
  vertices.push_back(edges.size());

  const auto all_vertices = graph.getVertices();
  for (const auto& i : all_vertices) {
    const auto& c_edges = graph.getEdges(i);
    edges.insert(edges.end(), c_edges.begin(), c_edges.end());
    vertices.push_back(edges.size());
  }

  // Use PMC to calculate
  pmc::pmc_graph G(vertices, edges);
  return solvePmc(G, params_, nullptr, nullptr, nullptr);
}

std::vector<int> teaser::MaxCliqueSolver::findMaxCliqueImpl(
    const teaser::Graph& graph, const std::vector<int>* previous_clique,
    detail::PmcHeuristic* heuristic) {
  // Handle deprecated field
  if (!params_.solve_exactly) {
    params_.solver_mode = CLIQUE_SOLVER_MODE::PMC_HEU;
  }

  // Create a PMC graph from the TEASER graph
  std::vector<int> edges;
  std::vector<long long> vertices;
  const int vertex_count = graph.numVertices();
  edges.reserve(static_cast<std::size_t>(graph.numEdges()) * 2);
  vertices.reserve(static_cast<std::size_t>(vertex_count) + 1);
  vertices.push_back(0);

  for (int i = 0; i < vertex_count; ++i) {
    const auto& c_edges = graph.getEdges(i);
    edges.insert(edges.end(), c_edges.begin(), c_edges.end());
    vertices.push_back(edges.size());
  }

  // Use PMC to calculate
  pmc::pmc_graph G(std::move(vertices), std::move(edges));
  return solvePmc(G, params_, &graph, previous_clique, heuristic);
}
