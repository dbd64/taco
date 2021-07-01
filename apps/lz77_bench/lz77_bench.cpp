#include "../test/test.h"
#include "taco/tensor.h"
#include <benchmark/benchmark.h>
#include "../../src/lower/iteration_graph.h"
#include "taco.h"

#include <random>

using namespace taco;

#ifndef SEED
#define SEED 0
#endif
std::default_random_engine gen(SEED);

const Format dv({Dense});
const Format lz77f({LZ77});

Index makeLZ77Index(const std::vector<int>& rowptr, const std::vector<int>& dist,
                    const std::vector<int>& runs) {
  return Index(lz77f, {ModeIndex({makeArray(rowptr), makeArray(dist), makeArray(runs)})});
}

template<typename T>
TensorBase makeLZ77(const std::string& name, const std::vector<int>& dimensions,
                    const std::vector<int>& pos,
                    const std::vector<int>& dist,
                    const std::vector<int>& runs,
                    const std::vector<T>& vals) {
  taco_uassert(dimensions.size() == 1);
  Tensor<T> tensor(name, dimensions, {LZ77});
  auto storage = tensor.getStorage();
  storage.setIndex(makeLZ77Index(pos, dist, runs));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

Func getCopyFunc(){
  auto copyFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], 0);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      return Union(l, Complement(l));
  };
  Func copy("copy_", copyFunc, algFunc);
  return copy;
}

Func getPlusFunc(){
  auto plusFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], v[1]);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      auto r = Region(v[1]);
      return Union(Union(l, r), Union(Complement(l), Complement(r)));
  };
  Func plus_("plus_", plusFunc, algFunc);
  return plus_;
}

template <typename T = uint8_t>
Tensor<T>
gen_random_lz77(std::string name, int size, double uncompressed_threshold,
                int lower_dist, int upper_dist,
                int lower_runs, int upper_runs,
                int lower_vals, int upper_vals) {
  std::uniform_int_distribution<int> unif_dist(lower_dist, upper_dist);
  std::uniform_int_distribution<int> unif_runs(lower_runs, upper_runs);
  std::uniform_int_distribution<int> unif_vals(lower_vals, upper_vals);
  std::uniform_real_distribution<double> unif_compressed(0,1);

  std::vector<int> dist;
  std::vector<int> runs;
  std::vector<T> vals;

  int numRemaining = size-1;
  while(numRemaining > 0){
    vals.push_back((T) unif_vals(gen));
    if (unif_compressed(gen) > uncompressed_threshold) {
      auto run = std::min(unif_runs(gen), numRemaining-1);
      auto dist_v = std::min(unif_dist(gen), (int) vals.size());
      if (run == 0) dist_v = 0;
      dist.push_back(dist_v);
      runs.push_back(run);
      numRemaining -= run+1;
    } else {
      dist.push_back(0);
      runs.push_back(0);
      numRemaining -= 1;
    }
  }

  vals.push_back((T) unif_vals(gen));
  dist.push_back(0);
  runs.push_back(0);

  std::vector<int> pos = {0, static_cast<int>(vals.size())};
  return makeLZ77("lz77_"+name, {size}, pos, dist, runs, vals);
}

constexpr int size_lower = 10;
constexpr int size_upper = 100'000'000;
constexpr int size_mult = 10;

int numRandTensors = 0;
constexpr int minElements = 10'000'000;

static void CustomArguments(benchmark::internal::Benchmark *b) {
  for (int size = size_lower; size <= size_upper; size *= size_mult) { // Size of vector
    for (int run = 1'000; run <= 1'000'000; run *= 10) {
      b->Args({size, 0, run, true});
      for (int thresh = 0; thresh <= 10; thresh += 5) {
        b->Args({size, thresh, run, false});
      }
    }
  }
}


static void BM_all(benchmark::State &state) {
  int tsize = state.range(0);
  double thresh = state.range(1)/10.0;
  int run_upper = state.range(2);
  bool isDense = state.range(3);
  auto plus_ = getPlusFunc();
  auto copy = getCopyFunc();

  auto d0 = gen_random_lz77("t0", tsize, thresh, 1, 50, run_upper-100, run_upper, 0, 255);
  auto d1 = gen_random_lz77("t1", tsize, thresh, 1, 50, run_upper-100, run_upper, 0, 255);

  if (isDense){
    Tensor<uint8_t> t0("t0_d", {tsize},   dv, 0);
    Tensor<uint8_t> t1("t1_d", {tsize},   dv, 0);
    const IndexVar i("i");

    t0(i) = copy(d0(i));
    t0.setAssembleWhileCompute(true);
    t0.compile();
    t0.compute();
    d0 = t0;

    t1(i) = copy(d1(i));
    t1.setAssembleWhileCompute(true);
    t1.compile();
    t1.compute();
    d1 = t1;
  }

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<uint8_t> expected("expected_", {tsize}, isDense ? dv : lz77f);
    const IndexVar i("i");
    expected(i) = plus_(d0(i), d1(i));
    expected.setAssembleWhileCompute(true);
    expected.compile();
    state.ResumeTiming();

    expected.compute();
  }
}

BENCHMARK(BM_all)->Apply(CustomArguments)\
                 ->MeasureProcessCPUTime()\
                 ->Unit(benchmark::kMicrosecond)\
                 ->Repetitions(10);

BENCHMARK_MAIN();