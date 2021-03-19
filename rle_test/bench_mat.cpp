#include "../test/test.h"
//#include "test_tensors.h"
#include <benchmark/benchmark.h>
#include "taco/tensor.h"

#include <random>

using namespace taco;

const Format dv({Dense, Dense});
const Format rv({Dense, RLE});
const IndexVar i("i"), j("j"), k("k");

template <typename T=double>
std::pair<Tensor<T>, Tensor<T>> gen_random_rle(std::string name,
                                               int sizer= 100, int sizec=100,
                                               int lower_rle=1, int upper_rle=512,
                                               int lower=0, int upper=1) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<T> unif_vals(lower, upper);

  if (name.empty()){ name = "_"; }
  else { name = "_" + name + "_"; }

  Tensor<double> r("r" + name, {sizer, sizec}, rv);
  for (int x=0; x< sizer; x++) {
    int index = 0;
    while (index < sizec) {
      int numCopies = min(unif_rle(gen), sizec - index);
      T val = unif_vals(gen);
      for (int i = 0; i < numCopies; i++) {
        r.insert({x, index + i}, val);
      }
      index += numCopies;
    }
  }
  r.pack();

  Tensor<double> d("d"+name, {sizer, sizec}, dv);
  d(i,j) = r(i,j);
  d.evaluate();

  return {r,d};
}
std::vector<std::pair<int,int>> rle_ranges =
        {{1,512}, {1,1024}, {512,1024}, {1000,10000},
         {10000,100000}, {1, 10}};
constexpr int size_lower = 50;
constexpr int size_upper = 1600;
constexpr int size_mult = 2;
constexpr int val_lower = 1;
constexpr int val_upper = 100;
constexpr int val_mult = 10;

constexpr int numRandTensors = 20;



static void CustomArguments(benchmark::internal::Benchmark* b) {
  for(int i=size_lower; i<=size_upper; i*=size_mult) { // Size of vector
    for (int j = val_lower; j <= val_upper; j*=val_mult) { // Upper limit of values
      for (auto &x : rle_ranges) { // bounds on rle values
        auto[l, u] = x;
        b->Args({i, j, l, u});
      }
    }
  }
}

using vpTensor = std::vector<std::pair<Tensor<double>, Tensor<double>>>;
std::map<std::tuple<int,int,int,int>, vpTensor> ts;

static void BM_dense_sum(benchmark::State& state) {
  int tsize = state.range(0);
  int upperVal = state.range(1);
  int lRle = state.range(2);
  int uRle = state.range(3);

  auto& res = ts[{tsize, upperVal, lRle, uRle}];
  for (auto _ : state) {
    for(int x=0; x<numRandTensors; x++) {
      auto [r0, d0] = res[2*x];
      auto [r1, d1] = res[2*x+1];
      Tensor<double> expected("expected_", {tsize,tsize}, dv);
      expected(i,j) = d0(i,j) + d1(i,j);
      expected.evaluate();
    }
  }
}

BENCHMARK(BM_dense_sum)->Apply(CustomArguments);


static void BM_rle_sum(benchmark::State& state) {
  int tsize = state.range(0);
  int upperVal = state.range(1);
  int lRle = state.range(2);
  int uRle = state.range(3);

  auto& res = ts[{tsize, upperVal, lRle, uRle}];
  for (auto _ : state) {
    for(int x=0; x<numRandTensors; x++) {
      auto [r0, d0] = res[2*x];
      auto [r1, d1] = res[2*x+1];

      Tensor<double> expected("expected_", {tsize,tsize}, dv);
      expected(i,j) = r0(i,j) + r1(i,j);
      expected.evaluate();
    }
  }
}
BENCHMARK(BM_rle_sum)->Apply(CustomArguments);

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  for(int i=size_lower; i<=size_upper; i*=size_mult) { // Size of vector
    for (int j = val_lower; j <= val_upper; j*=val_mult) { // Upper limit of values
      for (auto &x : rle_ranges) { // bounds on rle values
        auto [l, u] = x;
        vpTensor vs;
        for (int k=0; k< numRandTensors; k++) {
          std::cout << "loop : " << i <<", " << j << ", " << l << ", " << u << std::endl;

          auto[r0, d0] = gen_random_rle<double>("0", i, i, l, u, 0, j);
          auto[r1, d1] = gen_random_rle<double>("1", i, i, l, u, 0, j);
          vs.push_back({r0, d0});
          vs.push_back({r1, d1});
        }
        ts.insert({{i,j, l, u}, vs});
      }
    }
  }

  ::benchmark::RunSpecifiedBenchmarks();
}
