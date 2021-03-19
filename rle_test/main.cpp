#include "../test/test.h"
//#include "test_tensors.h"
#include <benchmark/benchmark.h>
#include "taco/tensor.h"

#include <random>

using namespace taco;

const Format dv({Dense});
const Format rv({RLE});
const IndexVar i("i"), j("j"), k("k");

template <typename T=double, int size=100>
Tensor<T> gen_random_dense(std::string name, int lower=0, int upper=1) {
  std::default_random_engine gen;
  std::uniform_int_distribution<T> unif(lower, upper);

  if (name.empty()){ name = "_"; }
  else { name = "_" + name + "_"; }

  Tensor<double> d("d" + name, {size}, dv);
  for (int i = 0; i < d.getDimension(0); ++i) {
    d.insert({i}, unif(gen));
  }
  d.pack();

  return d;
}


template <typename T=double, int size=100>
std::pair<Tensor<T>, Tensor<T>> gen_random(std::string name="", int lower=0, int upper=1) {
  Tensor<T> d = gen_random_dense<T,size>(name, lower, upper);

  if (name.empty()){ name = "_"; }
  else { name = "_" + name + "_"; }

  Tensor<double> r("r" + name, {size}, rv);
  r(i) = d(i);
  r.setAssembleWhileCompute(true);
  r.compile();
  r.compute();

  return {r,d};
}

template <typename T=double>
std::pair<Tensor<T>, Tensor<T>> gen_random_rle(std::string name, int size= 100, int lower_rle=1, int upper_rle=512,
                                               int lower=0, int upper=1) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<T> unif_vals(lower, upper);

  if (name.empty()){ name = "_"; }
  else { name = "_" + name + "_"; }

  Tensor<double> r("r" + name, {size}, rv);
  int index = 0;
  while (index < size){
    int numCopies = min(unif_rle(gen), size-index);
    T val = unif_vals(gen);
    for (int i=0; i< numCopies; i++){
      r.insert({index+i}, val);
    }
    index+=numCopies;
  }
  r.pack();

  Tensor<double> d("d"+name, {size}, dv);
  d(i) = r(i);
  d.evaluate();

  return {r,d};
}
std::vector<std::pair<int,int>> rle_ranges =
        {{1,512}, {1,1024}, {512,1024}, {1000,10000},
         {10000,100000}, {1, 10}};
constexpr int size_lower = 100;
constexpr int size_upper = 100'000;
constexpr int size_mult = 10;
constexpr int val_lower = 1;
constexpr int val_upper = 100;
constexpr int val_mult = 10;

constexpr int numRandTensors = 25;



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
    for(int j=0; j<numRandTensors; j++) {
      auto [r0, d0] = res[2*j];
      auto [r1, d1] = res[2*j+1];
      Tensor<double> expected("expected_", {tsize}, dv);
      expected(i) = d0(i) + d1(i);
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
    for(int j=0; j<numRandTensors; j++) {
      auto [r0, d0] = res[2*j];
      auto [r1, d1] = res[2*j+1];

      Tensor<double> expected("expected_", {tsize}, dv);
      expected(i) = r0(i) + r1(i);
      expected.evaluate();
    }
  }
}
BENCHMARK(BM_rle_sum)->Apply(CustomArguments);

int main(int argc, char** argv) {
  for(int i=size_lower; i<=size_upper; i*=size_mult) { // Size of vector
    for (int j = val_lower; j <= val_upper; j*=val_mult) { // Upper limit of values
      for (auto &x : rle_ranges) { // bounds on rle values
        auto [l, u] = x;
        vpTensor vs;
        for (int k=0; k< numRandTensors; k++) {
          std::cout << "loop : " << i <<", " << j << ", " << l << ", " << u << std::endl;

          auto[r0, d0] = gen_random_rle<double>("0", i, l, u, 0, j);
          auto[r1, d1] = gen_random_rle<double>("1", i, l, u, 0, j);
          vs.push_back({r0, d0});
          vs.push_back({r1, d1});
        }
        ts.insert({{i,j, l, u}, vs});
      }
    }
  }
  std::cout << "HELLO!" << std::endl;

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
}
