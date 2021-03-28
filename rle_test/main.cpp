#include "../test/test.h"
//#include "test_tensors.h"
#include "taco/tensor.h"
#include <benchmark/benchmark.h>

#include <random>

using namespace taco;

std::default_random_engine gen(0);


const Format dv({Dense});
const Format rv({RLE});
const IndexVar i("i"), j("j"), k("k");

template <typename T = double>
std::pair<Tensor<T>, Tensor<T>>
gen_random_rle(std::string name, int size = 100, int lower_rle = 1,
               int upper_rle = 512, int lower = 0, int upper = 1, int rle_size_bits = 8) {
  Format rv({RLE_s(rle_size_bits)});

  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<T> unif_vals(lower, upper);

  if (name.empty()) {
    name = "_";
  } else {
    name = "_" + name + "_";
  }

  Tensor<double> r("r" + name, {size}, rv);
  int index = 0;
  while (index < size) {
    int numCopies = min(unif_rle(gen), size - index);
    T val = unif_vals(gen);
    for (int i = 0; i < numCopies; i++) {
      r.insert({index + i}, val);
    }
    index += numCopies;
  }
  r.pack();

  Tensor<double> d("d" + name, {size}, dv);
  d(i) = r(i);
  d.evaluate();

  return {r, d};
}
std::vector<std::pair<int, int>> rle_ranges = //{{1,10}, {1,1024}, {1024,2048}};
    {{1, 512}, {1, 1024}, {512, 1024}, {1000, 10000}, {10000, 100000}, {1, 10}};
constexpr int size_lower = 10;
constexpr int size_upper = 1'000'000;
constexpr int size_mult = 10;
constexpr int val_lower = 1;
constexpr int val_upper = 100;
constexpr int val_mult = 100;
constexpr int rle_bits_lower = 16;
constexpr int rle_bits_upper = 32;
constexpr int rle_bits_mult = 2;

constexpr int numRandTensors = 1;

static void CustomArguments(benchmark::internal::Benchmark *b) {
  for(int r = rle_bits_lower; r <= rle_bits_upper; r*= rle_bits_mult) {
    for (int i = size_lower; i <= size_upper; i *= size_mult) { // Size of vector
      for (int j = val_lower; j <= val_upper; j *= val_mult) {          // Upper limit of values
        for (auto &x : rle_ranges) { // bounds on rle values
          auto[l, u] = x;
          b->Args({i, j, l, u, r, true});
          b->Args({i, j, l, u, r, false});
        }
      }
    }
  }
}

using vpTensor = std::vector<std::pair<Tensor<double>, Tensor<double>>>;

vpTensor current;

bool isFilled;
std::map<std::tuple<int, int, int, int, int>, vpTensor> ts;

vpTensor& getCurrent(int tsize, int upperVal, int lRle, int uRle, int r, bool isDense){
  assert(isDense == !isFilled);
  if(!isFilled){
    vpTensor vs;
    for (int k = 0; k < numRandTensors; k++) {
      auto[r0, d0] = gen_random_rle<double>("0", tsize, lRle, uRle, 0, upperVal, r);
      auto[r1, d1] = gen_random_rle<double>("1", tsize, lRle, uRle, 0, upperVal, r);
      vs.push_back({r0, d0});
      vs.push_back({r1, d1});
    }
    current = vs;
  }
  isFilled = !isFilled;
  return current;
}

static void BM_all(benchmark::State &state) {
  int tsize = state.range(0);
  int upperVal = state.range(1);
  int lRle = state.range(2);
  int uRle = state.range(3);
  int r = state.range(4);
  bool isDense = state.range(5);
  Format rv({RLE_s(r)});

  auto& res = getCurrent(tsize, upperVal, lRle, uRle, r ,isDense);

  std::vector<Tensor<double>> es(numRandTensors);
  for (int j = 0; j < numRandTensors; j++) {
    state.PauseTiming();
    auto [r0, d0] = res[2 * j];
    auto [r1, d1] = res[2 * j + 1];
    Tensor<double> expected("expected_", {tsize}, isDense? dv : rv);
    expected(i) = d0(i) + d1(i);
    expected.compile();
    expected.assemble();
    es.push_back(expected);
  }

  for (auto _ : state) {
    for (auto& expected : es) {
      expected.computeAndClear();
    }
  }
}

BENCHMARK(BM_all)->Apply(CustomArguments);

BENCHMARK_MAIN();