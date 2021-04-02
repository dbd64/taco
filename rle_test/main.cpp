#include "../test/test.h"
//#include "test_tensors.h"
#include "taco/tensor.h"
#include <benchmark/benchmark.h>

#include <random>

using namespace taco;

std::default_random_engine gen(0);


const Format dv({Dense});

template <typename T = double>
Tensor<T>
gen_random_rle(std::string name, int size = 100, int lower_rle = 1,
               int upper_rle = 512, int lower = 0, int upper = 1) {
  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<T> unif_vals(lower, upper);

  if (name.empty()) {
    name = "_";
  } else {
    name = "_" + name + "_";
  }

  Tensor<double> r("r" + name, {size}, dv);
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
  return r;
}

std::vector<std::pair<int, int>> rle_ranges = //{{1,10}, {1,1024}, {1024,2048}};
    {{1, 10}, {1, 1000}, {1000, 10000}};
constexpr int size_lower = 10;
constexpr int size_upper = 10'000'000;
constexpr int size_mult = 10;
constexpr int val_lower = 1;
constexpr int val_upper = 100;
constexpr int val_mult = 100;
constexpr int rle_bits_lower = 8;
constexpr int rle_bits_upper = 32;
constexpr int rle_bits_mult = 2;

constexpr int numRandTensors = 100;

static void CustomArguments(benchmark::internal::Benchmark *b) {
  for (int i = size_lower; i <= size_upper; i *= size_mult) { // Size of vector
    for (int j = val_lower; j <= val_upper; j *= val_mult) {          // Upper limit of values
      for (auto &x : rle_ranges) { // bounds on rle values
        auto[l, u] = x;
        b->Args({i, j, l, u, 0});
        for(int r = rle_bits_lower; r <= rle_bits_upper; r*= rle_bits_mult) {
          b->Args({i, j, l, u, r});
        }
      }
    }
  }
}

using vpTensor = std::vector<Tensor<double>>;
vpTensor current;
vpTensor& updateCurrent(int tsize, int upperVal, int lRle, int uRle, bool isDense){
  if(isDense){
    vpTensor vs;
    for (int k = 0; k < numRandTensors; k++) {
      auto d0 = gen_random_rle<double>("0", tsize, lRle, uRle, 0, upperVal);
      auto d1 = gen_random_rle<double>("1", tsize, lRle, uRle, 0, upperVal);

      vs.push_back(d0);
      vs.push_back(d1);
    }
    current = vs;
  }
  return current;
}

vpTensor getCurrentR(int tsize, int upperVal, int lRle, int uRle, int bits) {
  if(bits == 0) {
    return current;
  }

  if(current.size() == 0){
    std::cout << "WARNING!" << std::endl;
    updateCurrent(tsize, upperVal, lRle, uRle, true);
  }

  Format rv({RLE_s(bits, tsize)});

  vpTensor vs;
  bool toggle = true;
  std::string name0("r0");
  std::string name1("r1");

  for(auto& d : current){
    std::string name = toggle? name0: name1;
    toggle = !toggle;
    Tensor<double> r(name, {tsize}, rv);
    IndexVar i("i");
    r(i) = d(i);
    r.evaluate();
    compress_rle_b<double>(r, bits);

    vs.push_back(r);
  }

  return vs;
}

static void BM_all(benchmark::State &state) {
  int tsize = state.range(0);
  int upperVal = state.range(1);
  int lRle = state.range(2);
  int uRle = state.range(3);
  int bits = state.range(4);
  bool isDense = bits == 0;
  Format rv({RLE_s(bits, tsize)});

  updateCurrent(tsize, upperVal, lRle, uRle, isDense);
  auto res = getCurrentR(tsize, upperVal, lRle, uRle, bits);

  for (auto _ : state) {
    for (int j = 0; j < numRandTensors; j++) {
      state.PauseTiming();
      auto d0 = res[2 * j];
      auto d1 = res[2 * j + 1];
      Tensor<double> expected("expected_", {tsize}, isDense ? dv : rv);
      const IndexVar i("i");
      expected(i) = d0(i) + d1(i);
      if (!isDense) {
        expected.setAssembleWhileCompute(true);
      }
      expected.compile();
      expected.assemble();
      state.ResumeTiming();
      expected.compute();
    }
  }
}

BENCHMARK(BM_all)->Apply(CustomArguments)->ArgNames({"size", "value_upper", "rle_lower", "rle_upper", "rle_bits"})->MeasureProcessCPUTime()->Unit(benchmark::kMicrosecond); //->Repetitions(10)->ReportAggregatesOnly(false)->DisplayAggregatesOnly(false);

BENCHMARK_MAIN();