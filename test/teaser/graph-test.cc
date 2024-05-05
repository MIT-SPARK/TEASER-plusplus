/**
 * Copyright 2020, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Jingnan Shi, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 */

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <iostream>
#include <map>

#include "pmc/pmc.h"
#include "pmc/pmc_input.h"
#include "teaser/graph.h"
#include "test_utils.h"

/**
 * A helper function to generate mock input for testing PMC code
 * @return a mock input to PMC max clique functions
 */
pmc::input generateMockInput() {
  pmc::input in;
  in.algorithm = 0;
  in.threads = 12;
  in.experiment = 0;
  in.lb = 0;
  in.ub = 0;
  in.param_ub = 0;
  in.adj_limit = 20000;
  in.time_limit = 3600;
  in.remove_time = 4;
  in.graph_stats = false;
  in.verbose = false;
  in.help = false;
  in.MCE = false;
  in.decreasing_order = false;
  in.heu_strat = "kcore";
  in.vertex_search_order = "deg";
  return in;
}

/**
 * Print adjacency matrix given a bool**
 * @param adj a bool** to the adjcency matrix
 * @param nodes number of nodes in the graph
 */
void printAdjMatirx(const std::vector<std::vector<bool>>& adj, int nodes) {
  std::cout << "Adjacency matrix: " << std::endl;
  for (auto& i : adj) {
    for (auto j : i) {
      std::cout << j << " ";
    }
    std::cout << std::endl;
  }
}

TEST(GraphTest, BasicFunctions) {
  {
    // Test Basic Member Functions
    teaser::Graph graph;
    graph.populateVertices(2);
    graph.addEdge(1, 0);
    EXPECT_TRUE(graph.hasEdge(1, 0));
    EXPECT_TRUE(graph.hasVertex(1));
    EXPECT_TRUE(graph.hasVertex(0));
    EXPECT_EQ(graph.numEdges(), 1);

    graph.addVertex(2);
    EXPECT_TRUE(graph.hasVertex(2));
    graph.addVertex(2);
    EXPECT_TRUE(graph.hasVertex(2));
    graph.addVertex(3);
    EXPECT_TRUE(graph.hasVertex(3));
    EXPECT_EQ(graph.numVertices(), 4);
  }
  {
    // A simple 3 vertex graph:
    // 1--2
    // 2--3
    // 3--1
    teaser::Graph graph;
    graph.populateVertices(4);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    graph.addEdge(3, 1);
    EXPECT_EQ(graph.numEdges(), 3);
    EXPECT_EQ(graph.numVertices(), 4);

    graph.removeEdge(1, 2);
    graph.removeEdge(2, 3);
    graph.removeEdge(1, 3);
    EXPECT_EQ(graph.numEdges(), 0);
    EXPECT_EQ(graph.numVertices(), 4);

    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    graph.addEdge(3, 1);
    EXPECT_EQ(graph.numEdges(), 3);
    EXPECT_EQ(graph.numVertices(), 4);
  }
  {
    // Use adjcency list to initialize
    std::map<int, std::vector<int>> vertices_map;

    // create a complete graph
    int nodes_count = 5;
    for (int i = 0; i < nodes_count; ++i) {
      std::vector<int> temp;
      for (int j = 0; j < nodes_count; ++j) {
        if (j != i) {
          temp.push_back(j);
        }
      }
      vertices_map[i] = temp;
    }

    teaser::Graph graph(vertices_map);

    auto adj_mat = graph.getAdjMatrix();
    std::cout << "Adj matrix for a complete graph: " << std::endl;
    std::cout << adj_mat << std::endl;

    EXPECT_EQ(graph.numVertices(), nodes_count);
    EXPECT_EQ(graph.numEdges(), nodes_count * (nodes_count - 1) / 2);
  }
}

TEST(PMCTest, FindMaximumClique1) {
  // A complete graph with max clique # = 5
  auto in = generateMockInput();
  std::map<int, std::vector<int>> vertices_map;

  // create a complete graph
  int nodes_count = 5;
  for (int i = 0; i < nodes_count; ++i) {
    std::vector<int> temp;
    for (int j = 0; j < nodes_count; ++j) {
      if (j != i) {
        temp.push_back(j);
      }
    }
    vertices_map[i] = temp;
  }
  pmc::pmc_graph G(vertices_map);
  G.create_adj();
  printAdjMatirx(G.adj, nodes_count);

  EXPECT_EQ(G.max_degree, nodes_count - 1);
  EXPECT_EQ(G.min_degree, nodes_count - 1);

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

  EXPECT_EQ(in.lb, 5);
  EXPECT_EQ(in.lb, in.ub);

  if (G.num_vertices() < in.adj_limit) {
    G.create_adj();
    pmc::pmcx_maxclique finder(G, in);
    finder.search_dense(G, C);
  } else {
    pmc::pmcx_maxclique finder(G, in);
    finder.search(G, C);
  }

  EXPECT_EQ(C.size(), 5);
}

TEST(PMCTest, FindMaximumClique2) {
  // A simple graph with max clique # = 3
  auto in = generateMockInput();
  std::map<int, std::vector<int>> vertices_map;

  // create a complete graph
  int nodes_count = 4;
  vertices_map[0] = {2, 3};
  vertices_map[1] = {2};
  vertices_map[2] = {0, 1, 3};
  vertices_map[3] = {0, 2};

  pmc::pmc_graph G(vertices_map);
  G.create_adj();
  printAdjMatirx(G.adj, nodes_count);

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

  EXPECT_EQ(in.lb, 3);
  EXPECT_EQ(in.lb, in.ub);

  if (G.num_vertices() < in.adj_limit) {
    G.create_adj();
    pmc::pmcx_maxclique finder(G, in);
    finder.search_dense(G, C);
  } else {
    pmc::pmcx_maxclique finder(G, in);
    finder.search(G, C);
  }

  EXPECT_EQ(C.size(), 3);
}

TEST(PMCTest, FindMaximumClique3) {
  // A graph with all disjoint nodes
  auto in = generateMockInput();
  std::map<int, std::vector<int>> vertices_map;

  // create a complete graph
  int nodes_count = 4;
  for (int i = 0; i < nodes_count; ++i) {
    vertices_map[i] = {};
  }

  pmc::pmc_graph G(vertices_map);
  G.create_adj();
  printAdjMatirx(G.adj, nodes_count);

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
  EXPECT_GE(in.ub, in.lb);

  // for isolated vertices, pmc has lb = 0, which will
  // cause a potential segfault in search dense function
  if (in.lb == 0) {
    in.lb += 1;
  }

  if (G.num_vertices() < in.adj_limit) {
    G.create_adj();
    pmc::pmcx_maxclique finder(G, in);
    finder.search_dense(G, C);
  } else {
    pmc::pmcx_maxclique finder(G, in);
    finder.search(G, C);
  }

  EXPECT_EQ(C.size(), 1);
}

TEST(MaxCliqueSolverTest, FindMaxClique) {
  // A complete graph with max clique # = 5
  {
    std::map<int, std::vector<int>> vertices_map;

    // create a complete graph
    int nodes_count = 5;
    for (int i = 0; i < nodes_count; ++i) {
      std::vector<int> temp;
      for (int j = 0; j < nodes_count; ++j) {
        if (j != i) {
          temp.push_back(j);
        }
      }
      vertices_map[i] = temp;
    }

    teaser::Graph graph(vertices_map);
    teaser::MaxCliqueSolver cliqueSolver;
    auto clique = cliqueSolver.findMaxClique(graph);
    EXPECT_EQ(clique.size(), 5);

    // Check whether the clique has the correct vertices
    std::cout << "Clique Nodes: ";
    std::copy(clique.begin(), clique.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::set<int> s(clique.begin(), clique.end());
    std::vector<int> ref_clique{0, 1, 2, 3, 4};
    for (const auto& i : ref_clique) {
      EXPECT_TRUE(s.find(i) != s.end());
    }
  }
}

TEST(PMCTest, FindMaximumCliqueSingleThreaded) {
  // A complete graph with max clique # = 5
  auto in = generateMockInput();
  in.threads = 1;
  std::map<int, std::vector<int>> vertices_map;

  // create a complete graph
  int nodes_count = 5;
  for (int i = 0; i < nodes_count; ++i) {
    std::vector<int> temp;
    for (int j = 0; j < nodes_count; ++j) {
      if (j != i) {
        temp.push_back(j);
      }
    }
    vertices_map[i] = temp;
  }
  pmc::pmc_graph G(vertices_map);
  G.create_adj();
  printAdjMatirx(G.adj, nodes_count);

  EXPECT_EQ(G.max_degree, nodes_count - 1);
  EXPECT_EQ(G.min_degree, nodes_count - 1);

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

  EXPECT_EQ(in.lb, 5);
  EXPECT_EQ(in.lb, in.ub);

  if (G.num_vertices() < in.adj_limit) {
    G.create_adj();
    pmc::pmcx_maxclique finder(G, in);
    finder.search_dense(G, C);
  } else {
    pmc::pmcx_maxclique finder(G, in);
    finder.search(G, C);
  }

  EXPECT_EQ(C.size(), 5);
}

TEST(PMCTest, FindMaximumCliqueMultiThreaded) {
  // A complete graph with max clique # = 5
  auto in = generateMockInput();
  in.threads = 15;
  std::map<int, std::vector<int>> vertices_map;

  // create a complete graph
  int nodes_count = 5;
  for (int i = 0; i < nodes_count; ++i) {
    std::vector<int> temp;
    for (int j = 0; j < nodes_count; ++j) {
      if (j != i) {
        temp.push_back(j);
      }
    }
    vertices_map[i] = temp;
  }
  pmc::pmc_graph G(vertices_map);
  G.create_adj();
  printAdjMatirx(G.adj, nodes_count);

  EXPECT_EQ(G.max_degree, nodes_count - 1);
  EXPECT_EQ(G.min_degree, nodes_count - 1);

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

  EXPECT_EQ(in.lb, 5);
  EXPECT_EQ(in.lb, in.ub);

  if (G.num_vertices() < in.adj_limit) {
    G.create_adj();
    pmc::pmcx_maxclique finder(G, in);
    finder.search_dense(G, C);
  } else {
    pmc::pmcx_maxclique finder(G, in);
    finder.search(G, C);
  }

  EXPECT_EQ(C.size(), 5);
}
