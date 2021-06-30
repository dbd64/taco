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

  int numRemaining = size;
  while(numRemaining > 0){
    if (unif_compressed(gen) > uncompressed_threshold) {
      auto run = std::min(unif_runs(gen), numRemaining);
      dist.push_back(unif_dist(gen));
      runs.push_back(run);
      vals.push_back((T) unif_vals(gen));
      numRemaining -= run;
    } else {
      dist.push_back(0);
      runs.push_back(0);
      vals.push_back((T) unif_vals(gen));
      numRemaining -= 1;
    }
  }

  std::vector<int> pos = {0, static_cast<int>(vals.size())};
  return makeLZ77("lz77_"+name, {size}, pos, dist, runs, vals);
}

constexpr int size_lower = 1'000;
constexpr int size_upper = 100'000'000;
constexpr int size_mult = 10;

int numRandTensors = 0;
constexpr int minElements = 10'000'000;

static void CustomArguments(benchmark::internal::Benchmark *b) {
  for (int size = size_lower; size <= size_upper; size *= size_mult) { // Size of vector
    for (int thresh = 0; thresh <= 10; thresh += 1) {
      b->Args({size, thresh, false});
//      b->Args({size, thresh, true});
    }
  }
}

using vpTensor = std::vector<Tensor<uint8_t>>;
vpTensor current;
vpTensor& updateCurrent(int tsize, double thresh, bool isDense){
  if(!isDense){
    vpTensor vs;
    numRandTensors = std::max(minElements/tsize, 1);
    for (int k = 0; k < numRandTensors; k++) {
      auto d0 = gen_random_lz77("t0", tsize, thresh, 1, 1'000, 1, 10'000, 0, 255);
      auto d1 = gen_random_lz77("t1", tsize, thresh, 1, 1'000, 1, 10'000, 0, 255);
      vs.push_back(d0);
      vs.push_back(d1);
    }
    current.swap(vs);
  }
  return current;
}

std::string name0("r0");
std::string name1("r1");
std::string name2("s0");
std::string name3("s1");
bool name_toggle = false;

vpTensor& getCurrentR(int tsize, double thresh, bool isDense) {
  if(!isDense) {
    return current;
  }

  if(current.empty()){
    std::cout << "WARNING!" << std::endl;
    updateCurrent(tsize, thresh, false);
  }

  vpTensor vs;
  bool toggle = true;
  auto copy = getCopyFunc();

  for(auto& d : current){
    std::string name = name_toggle ? (toggle? name0: name1) : (toggle? name2: name3);
    toggle = !toggle;
    Tensor<uint8_t> r(name, {tsize}, dv);
    IndexVar i("i");
    r(i) = copy(d(i));
    r.setAssembleWhileCompute(true);
    r.compile();
    r.compute();
    vs.push_back(r);
  }

  name_toggle = !name_toggle;
  current.swap(vs);

  return current;
}

//static void BM_all(benchmark::State &state) {
//  int tsize = state.range(0);
//  double thresh = state.range(1)/10.0;
//  bool isDense = state.range(2);
//  auto plus_ = getPlusFunc();
//
//  updateCurrent(tsize, thresh, isDense);
//  auto res = getCurrentR(tsize, thresh, isDense);
//  int numTensors = res.size()/2;
//
//  for (auto _ : state) {
//    for (int j = 0; j < numTensors; j++) {
//      state.PauseTiming();
//      auto d0 = res[2 * j];
//      auto d1 = res[2 * j + 1];
//      Tensor<uint8_t> expected("expected_", {tsize}, isDense ? dv : lz77f);
//      const IndexVar i("i");
//      expected(i) = plus_(d0(i), d1(i));
//      expected.setAssembleWhileCompute(true);
//      expected.compile();
//      state.ResumeTiming();
//      expected.compute();
//    }
//  }
//
//  int t0_vals_size_total = 0;
//  int t1_vals_size_total = 0;
//  for (int j = 0; j < numTensors; j++) {
//    state.PauseTiming();
//    auto d0 = res[2 * j];
//    auto d1 = res[2 * j + 1];
//    if(isDense){
//      t0_vals_size_total += tsize;
//      t1_vals_size_total += tsize;
//    } else {
//      taco_tensor_t *t0 = d0.getStorage();
//      t0_vals_size_total += ((int32_t *)(t0->indices[0][0]))[1];
//
//      taco_tensor_t *t1 = d0.getStorage();
//      t1_vals_size_total += ((int32_t *)(t1->indices[0][0]))[1];
//    }
//  }
//
//  state.counters.insert({{"num_tensors", numTensors},
//                         {"t0_vals_size_total", t0_vals_size_total},
//                         {"t1_vals_size_total", t1_vals_size_total}});
//
//}

static void BM_all(benchmark::State &state) {
  int tsize = state.range(0);
  double thresh = state.range(1)/10.0;
  bool isDense = state.range(2);
  auto plus_ = getPlusFunc();

  // std::default_random_engine gen2(SEED);
  // gen = gen2;

  auto d0 = gen_random_lz77("t0", tsize, thresh, 1, 1'000, 1'000, 10'000, 0, 255);
  auto d1 = gen_random_lz77("t1", tsize, thresh, 1, 1'000, 1'000, 10'000, 0, 255);

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
//                 ->ArgNames({"size", "value_upper", "rle_lower", "rle_upper", "rle_bits"})\
;
//->Repetitions(10)->ReportAggregatesOnly(false)->DisplayAggregatesOnly(false);

BENCHMARK_MAIN();