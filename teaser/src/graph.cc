/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "teaser/graph.h"
#include "pmc/pmc.h"

vector<int> teaser::MaxCliqueSolver::findMaxClique(teaser::Graph graph) {

  // Create a PMC graph from the TEASER graph
  vector<int> edges;
  vector<long long> vertices;
  vertices.push_back(edges.size());

  const auto all_vertices = graph.getVertices();
  for (const auto& i : all_vertices) {
    const auto& c_edges = graph.getEdges(i);
    edges.insert(edges.end(), c_edges.begin(), c_edges.end());
    vertices.push_back(edges.size());
  }

  // Use PMC to calculate
  pmc::pmc_graph G(vertices, edges);

  // Prepare PMC input
  // TODO: Incorporate this to the constructor
  pmc::input in;
  in.algorithm = 0;
  in.threads = 12;
  in.experiment = 0;
  in.lb = 0;
  in.ub = 0;
  in.param_ub = 0;
  in.adj_limit = 20000;
  in.time_limit = params_.time_limit;
  in.remove_time = 4;
  in.graph_stats = false;
  in.verbose = false;
  in.help = false;
  in.MCE = false;
  in.decreasing_order = false;
  in.heu_strat = "kcore";
  in.vertex_search_order = "deg";

  // upper-bound of max clique
  G.compute_cores();
  if (in.ub == 0) {
    in.ub = G.get_max_core() + 1;
  }

  // lower-bound of max clique
  vector<int> C;
  if (in.lb == 0 && in.heu_strat != "0") { // skip if given as input
    pmc::pmc_heu maxclique(G, in);
    in.lb = maxclique.search(G, C);
  }

  assert(in.lb != 0);
  if (in.lb == 0) {
    // This means that max clique has a size of one
    TEASER_DEBUG_ERROR_MSG("Max clique lb equals to zero. Abort.");
    return C;
  }

  if (in.lb == in.ub) {
    return C;
  }

  // Optional exact max clique finding
  if (params_.solve_exactly) {
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
