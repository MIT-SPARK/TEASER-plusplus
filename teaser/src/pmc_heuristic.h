#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "pmc/pmc_heu.h"

namespace teaser {
namespace detail {

// Bitset implementation of the single-threaded k-core heuristic from Ryan
// Rossi's PMC. Keep PMC's seed order and candidate sort, including ties.
class PmcHeuristic {
public:
  void prepare(const pmc::pmc_graph& graph) {
    const auto& offsets = graph.get_vertices();
    const auto& edges = graph.get_edges();
    const int size = graph.num_vertices();
    words_ = (static_cast<std::size_t>(size) + 63) / 64;
    adjacency_.assign(size * words_, 0);
    common_.resize(words_);
    for (int u = 0; u < size; ++u) {
      auto* row = adjacency_.data() + u * words_;
      for (auto i = offsets[u]; i < offsets[u + 1]; ++i) {
        const int v = edges[i];
        row[v / 64] |= bit(v);
      }
    }
  }

  int search(const pmc::pmc_graph& graph, const pmc::pmc_heu& heuristic,
             std::vector<int>& clique) {
    const auto& offsets = graph.get_vertices();
    const auto& edges = graph.get_edges();
    const auto& cores = *heuristic.K;
    const auto& order = *heuristic.order;
    int best = 0;
    candidates_.clear();
    candidates_.reserve(graph.get_max_degree() + 1);
    clique.reserve(heuristic.ub);

    for (int i = graph.num_vertices() - 1; i >= 0 && best < heuristic.ub; --i) {
      const int root = order[i];
      if (cores[root] <= best) {
        continue;
      }
      for (auto j = offsets[root]; j < offsets[root + 1]; ++j) {
        const int v = edges[j];
        if (cores[v] > best) {
          candidates_.emplace_back(v, cores[v]);
        }
      }
      // Match PMC's handling of the candidate buffer even when it skips a seed.
      if (candidates_.size() <= static_cast<std::size_t>(best)) {
        continue;
      }

      // Greedy branching follows a single path. Each selected vertex
      // intersects the common neighborhood without reordering the candidates.
      std::sort(candidates_.begin(), candidates_.end(), pmc::pmc_heu::incr_heur);
      const int first = candidates_.back().get_id();
      std::copy(row(first), row(first) + words_, common_.begin());
      branch_.clear();
      branch_.push_back(first);
      for (auto it = candidates_.rbegin() + 1; it != candidates_.rend(); ++it) {
        const int v = it->get_id();
        if (!(common_[v / 64] & bit(v))) {
          continue;
        }
        branch_.push_back(v);
        const auto* neighbors = row(v);
        for (std::size_t w = 0; w < words_; ++w) {
          common_[w] &= neighbors[w];
        }
      }
      const int size = static_cast<int>(branch_.size()) + 1;
      if (size > best) {
        best = size;
        clique.assign(branch_.rbegin(), branch_.rend());
        clique.push_back(root);
      }
      candidates_.clear();
    }
    return best;
  }

private:
  using Word = std::uint64_t;

  static Word bit(int v) { return Word{1} << (v % 64); }
  const Word* row(int v) const { return adjacency_.data() + v * words_; }

  std::size_t words_ = 0;
  std::vector<Word> adjacency_;
  std::vector<Word> common_;
  std::vector<pmc::Vertex> candidates_;
  std::vector<int> branch_;
};

} // namespace detail
} // namespace teaser
