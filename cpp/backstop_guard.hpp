// The headline guard, one step at a time, in C++ for a robot's control loop.
//
// A port of `backstop.guard.online` and nothing more: the same score
// (`action_chunk_disagreement` on one step) and the same frozen
// `StopwatchRule`, read from the JSON that `scripts/freeze_guard.py` writes.
// Fitting the rule stays in Python. Header-only, C++17, no dependencies.
// `tests/unit/test_guard_cpp.py` checks it stops where the Python guard does.
//
// Reads only the K action samples and the step count. No sim state (ADR-003).
#pragma once

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace backstop {

// Only the two fields the guard reads; the rest of the file is provenance.
struct StopwatchRule {
  double line;
  long stop_step;
};

enum class Trigger { signal, clock };

inline const char* trigger_name(Trigger t) { return t == Trigger::signal ? "signal" : "clock"; }

// The `rule` object of a frozen rule file. Not a general JSON parser: it
// relies on the flat object of numbers that `save_rule` writes.
inline StopwatchRule load_rule(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open rule file " + path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  const auto begin = text.find("\"rule\"");
  const auto end = text.find('}', begin);
  if (begin == std::string::npos || end == std::string::npos)
    throw std::runtime_error(path + " has no \"rule\" object");
  const std::string rule = text.substr(begin, end - begin);
  auto number = [&](const std::string& key) {
    const auto at = rule.find("\"" + key + "\"");
    if (at == std::string::npos) throw std::runtime_error(path + ": rule has no " + key);
    return std::strtod(rule.c_str() + rule.find(':', at) + 1, nullptr);
  };
  return {number("line"), static_cast<long>(number("stop_step"))};
}

// (K, chunk, dim) samples for one step, row-major, index 0 the executed plan
// -> mean absolute deviation of the other K-1 from it, in double as Python
// does. Zero when K < 2: nothing to disagree with.
template <typename T>
double step_disagreement(const T* samples, std::size_t k, std::size_t chunk, std::size_t dim) {
  if (k < 2) return 0.0;
  const std::size_t plan = chunk * dim;
  double total = 0.0;
  for (std::size_t s = 1; s < k; ++s)
    for (std::size_t i = 0; i < plan; ++i) {
      const double d = static_cast<double>(samples[s * plan + i]) - static_cast<double>(samples[i]);
      total += d < 0 ? -d : d;
    }
  return total / static_cast<double>((k - 1) * plan);
}

class OnlineGuard {
 public:
  explicit OnlineGuard(StopwatchRule rule, long cooldown = 0) : rule_(rule), cooldown_(cooldown) {
    if (cooldown < 0) throw std::invalid_argument("cooldown must be >= 0");
  }

  // True when the guard stops the run at `step` (0-based, absolute). The
  // signal first, then the clock, as in Python.
  bool check(double score, long step) {
    fired_reason_.reset();
    if (step >= signal_muted_until_ && score >= rule_.line)
      fired_reason_ = Trigger::signal;
    else if (step - attempt_start_ >= rule_.stop_step)
      fired_reason_ = Trigger::clock;
    return fired_reason_.has_value();
  }

  template <typename T>
  bool check_samples(const T* samples, std::size_t k, std::size_t chunk, std::size_t dim,
                     long step) {
    return check(step_disagreement(samples, k, chunk, dim), step);
  }

  // Start a fresh attempt at `step`: restart the clock, mute the signal.
  void rearm(long step) {
    attempt_start_ = step;
    signal_muted_until_ = step + cooldown_;
    fired_reason_.reset();
  }

  std::optional<Trigger> fired_reason() const { return fired_reason_; }

 private:
  StopwatchRule rule_;
  long cooldown_;
  std::optional<Trigger> fired_reason_;
  long attempt_start_ = 0;
  long signal_muted_until_ = 0;
};

}  // namespace backstop
