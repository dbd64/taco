#include "../test/test.h"
#include "taco/tensor.h"
#include <benchmark/benchmark.h>
#include "../../src/lower/iteration_graph.h"
#include "taco.h"

#include <random>
#include <variant>

using namespace taco;

#ifndef SEED
#define SEED 0
#endif
std::default_random_engine gen(SEED);

const Format dv({Dense});
const Format lz77f({LZ77});

template <typename T>
union GetBytes {
    T value;
    uint8_t bytes[sizeof(T)];
};

using Repeat = std::pair<uint16_t, uint16_t>;

template <class T>
using TempValue = std::variant<T,Repeat>;

// helper type for the visitor #4
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename T>
void push_back(T arg, std::vector<uint8_t>& bytes, bool check = false){
  GetBytes<T> gb;
  gb.value = arg;

  if(check && ((gb.bytes[0] >> 7) & 0x1) == 1){
    bytes.push_back(1 << 7);
  }

  for (unsigned long i_=0; i_<sizeof(T); i_++){
    bytes.push_back(gb.bytes[i_]);
  }
}

template <typename T>
std::pair<std::vector<T>, int> packLZ77(std::vector<TempValue<T>> vals){
  std::vector<uint8_t> bytes;
  for (auto& val : vals){
    std::visit(overloaded {
            [&](T arg) { push_back(arg, bytes, true); },
            [&](std::pair<uint16_t, uint16_t> arg) {
                bytes.push_back(0x3 << 6);
                push_back(arg.first, bytes); push_back(arg.second, bytes);
            }
    }, val);
  }
  int size = bytes.size();
  while(bytes.size() % sizeof(T) != 0){
    bytes.push_back(0);
  }
  T* bytes_data = (T*) bytes.data();
  std::vector<T> values(bytes_data, bytes_data + (bytes.size() / sizeof(T)));

  return {values, size};
}

Index makeLZ77Index(const std::vector<int>& rowptr) {
  return Index(lz77f,
               {ModeIndex({makeArray(rowptr)})});
}

/// Factory function to construct a compressed sparse row (CSR) matrix.
template<typename T>
TensorBase makeLZ77(const std::string& name, const std::vector<int>& dimensions,
                    const std::vector<int>& pos, const std::vector<T>& vals) {
  taco_uassert(dimensions.size() == 1);
  Tensor<T> tensor(name, dimensions, {LZ77});
  auto storage = tensor.getStorage();
  storage.setIndex(makeLZ77Index(pos));
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

  std::vector<TempValue<T>> vals;
  int numRawValues = 1;
  vals.push_back((T) unif_vals(gen));

  int numRemaining = size-2;
  while(numRemaining > 0){
    if (unif_compressed(gen) > uncompressed_threshold && numRemaining > 1) {
      auto run = std::min(unif_runs(gen), numRemaining-1);
      auto dist_v = std::min(unif_dist(gen), numRawValues);
      if (run == 0) dist_v = 0;
      vals.push_back(Repeat{dist_v,run});
      numRemaining -= run;
    } else {
      vals.push_back((T) unif_vals(gen));
      numRawValues += 1;
      numRemaining -= 1;
    }
  }

  vals.push_back((T) unif_vals(gen));
  numRawValues += 1;

  auto packed = packLZ77<T>(vals);
  return makeLZ77<T>("lz77_"+name, {size}, {0, packed.second}, packed.first);
}

constexpr int size_lower = 100'000'000;
constexpr int size_upper = 100'000'000;
constexpr int size_mult  = 10;
constexpr int run_lower = 100'000;
constexpr int run_upper = 100'000; //1'000'000;
constexpr int run_mult  = 10;

int numRandTensors = 0;
constexpr int minElements = 10'000'000;

static void CustomArguments(benchmark::internal::Benchmark *b) {
  for (int size = size_lower; size <= size_upper; size *= size_mult) { // Size of vector
    for (int run = run_lower; run <= run_upper; run *= run_mult) {
      b->Args({size, 0, run, true});
      for (int thresh = 0; thresh <= 10; thresh += 15) {
        b->Args({size, thresh, run, false});
      }
    }
  }
}

static std::pair<Tensor<uint8_t>, Tensor<uint8_t>> getDense(Tensor<uint8_t> d0, Tensor<uint8_t> d1, int tsize){
  Tensor<uint8_t> t0("t0_d", {tsize},   dv, 0);
  Tensor<uint8_t> t1("t1_d", {tsize},   dv, 0);
  const IndexVar i("i");
  auto copy = getCopyFunc();

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

static void shim_compute(Tensor<uint8_t> t){
  t.compute();
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
    shim_compute(t0); //.compute();
    d0 = t0;

    t1(i) = copy(d1(i));
    t1.setAssembleWhileCompute(true);
    t1.compile();
    shim_compute(t1); //.compute();
    d1 = t1;
  }

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<uint8_t> expected("expected_", {tsize}, dv); //isDense ? dv : lz77f);
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
                 ->Unit(benchmark::kMicrosecond);
//                 ->Repetitions(10);

BENCHMARK_MAIN();