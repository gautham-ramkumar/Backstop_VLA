// Replays one recorded episode through the C++ guard, for parity checks.
//
//   guard_replay RULE_JSON COOLDOWN [REARM_STEP ...] < episode.bin
//
// stdin: four int64 (T, K, chunk, dim), then T*K*chunk*dim float64, row-major.
// stdout: one line per step, "<score> <signal|clock|->". The guard is rearmed
// just before each REARM_STEP is checked, as a recovery loop would.
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "backstop_guard.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s RULE_JSON COOLDOWN [REARM_STEP ...] < episode.bin\n", argv[0]);
    return 2;
  }
  try {
    backstop::OnlineGuard guard(backstop::load_rule(argv[1]), std::stol(argv[2]));
    std::set<long> rearms;
    for (int i = 3; i < argc; ++i) rearms.insert(std::stol(argv[i]));

    std::int64_t shape[4];
    if (std::fread(shape, sizeof shape, 1, stdin) != 1) throw std::runtime_error("no header");
    const auto [t, k, chunk, dim] = shape;
    std::vector<double> step(static_cast<std::size_t>(k * chunk * dim));
    for (long s = 0; s < t; ++s) {
      if (std::fread(step.data(), sizeof(double), step.size(), stdin) != step.size())
        throw std::runtime_error("episode ends early at step " + std::to_string(s));
      if (rearms.count(s)) guard.rearm(s);
      const double score = backstop::step_disagreement(step.data(), k, chunk, dim);
      const bool fired = guard.check(score, s);
      std::printf("%.17g %s\n", score, fired ? trigger_name(*guard.fired_reason()) : "-");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "guard_replay: %s\n", e.what());
    return 1;
  }
}
