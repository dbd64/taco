#include "../test/test.h"
//#include "test_tensors.h"
#include "taco/tensor.h"
#include <benchmark/benchmark.h>

#include <random>

using namespace taco;

std::default_random_engine gen(0);


const Format dv({Dense});
const Format rv({RLE});

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

static size_t unpackTensorData(const taco_tensor_t& tensorData,
                               const TensorBase& tensor) {
  auto storage = tensor.getStorage();
  auto format = storage.getFormat();

  std::vector<ModeIndex> modeIndices;
  size_t numVals = 1;
  for (int i = 0; i < tensor.getOrder(); i++) {
    ModeFormat modeType = format.getModeFormats()[i];
    if (modeType.getName() == Dense.getName()) {
      Array size = makeArray({*(int*)tensorData.indices[i][0]});
      modeIndices.push_back(ModeIndex({size}));
      numVals *= ((int*)tensorData.indices[i][0])[0];
    } else if (modeType.getName() == Sparse.getName()) {
      auto size = ((int*)tensorData.indices[i][0])[numVals];
      Array pos = Array(type<int>(), tensorData.indices[i][0], numVals+1, Array::UserOwns);
      Array idx = Array(type<int>(), tensorData.indices[i][1], size, Array::UserOwns);
      modeIndices.push_back(ModeIndex({pos, idx}));
      numVals = size;
    } else if (modeType.getName() == Singleton.getName()) {
      Array idx = Array(type<int>(), tensorData.indices[i][1], numVals, Array::UserOwns);
      modeIndices.push_back(ModeIndex({makeArray(type<int>(), 0), idx}));
    } else if (modeType.getName() == RLE.getName()) {
      auto valsSize = ((int*)tensorData.indices[i][0])[numVals];
      Array pos = Array(type<int>(), tensorData.indices[i][0], numVals+1, Array::UserOwns);
      Array rle = Array(type<uint16_t>(), tensorData.indices[i][1], valsSize, Array::UserOwns);
      modeIndices.push_back(ModeIndex({ pos, rle}));
      numVals = valsSize;
    } else {
      taco_not_supported_yet;
    }
  }
  storage.setIndex(Index(format, modeIndices));
  storage.setValues(Array(tensor.getComponentType(), tensorData.vals, numVals));
  return numVals;
}

template<class T, class R>
void compress_rle(Tensor<T>& tt) {
  taco_tensor_t* t = tt.getStorage();

  int max_rle = std::numeric_limits<R>::max();

  int32_t* t_pos = (int32_t*)(t->indices[0][0]);
  R* t_rle = (R*)(t->indices[0][1]);
  double* t_vals = (double*)(t->vals);

  double* t_vals_new = (double*)malloc(sizeof(double)*t_pos[1]);
  t_vals_new[0] = t_vals[0];

  int32_t tnpos = 0;
  double tnval = t_vals[0];

  for (int32_t itpos = 1; itpos < t_pos[1]; itpos++){
    double val = t_vals[itpos];
    if (val == tnval){
      if(t_rle[tnpos] == max_rle) {
        tnpos++;
        t_vals_new[tnpos] = tnval;
        t_rle[tnpos] = 1;
      } else {
        t_rle[tnpos]++;
      }
    } else {
      tnpos++;
      tnval = val;
      t_vals_new[tnpos] = tnval;
      t_rle[tnpos] = 1;
    }
  }

  void* res = realloc(t_vals_new, sizeof(double) * (tnpos+1));
  if(!res){ taco_uerror; }
  t->vals = (uint8_t*)res;

  res = realloc(t_rle, sizeof(R) * (tnpos+1));
  if(!res){ taco_uerror; }
  t->indices[0][1] = (uint8_t*)res;

  t_pos[1] = tnpos+1;

  tt.content->valuesSize = unpackTensorData(*t, tt);
}


std::vector<std::pair<int, int>> rle_ranges = //{{1,10}, {1,1024}, {1024,2048}};
    {{1, 10}, {1, 1024}, {1000, 10000}};
constexpr int size_lower = 10;
constexpr int size_upper = 10'000'000;
constexpr int size_mult = 10;
constexpr int val_lower = 1;
constexpr int val_upper = 100;
constexpr int val_mult = 100;
constexpr int rle_bits_lower = 8;
constexpr int rle_bits_upper = 64;
constexpr int rle_bits_mult = 2;

constexpr int numRandTensors = 50;

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

vpTensor getCurrentR(int tsize, int upperVal, int lRle, int uRle, int r) {
  if(r == 0) {
    return current;
  }

  if(current.size() == 0){
    std::cout << "WARNING!" << std::endl;
    updateCurrent(tsize, upperVal, lRle, uRle, true);
  }

  Format rv({RLE_s(r)});

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

    vs.push_back(r);
  }

  return vs;
}

static void BM_all(benchmark::State &state) {
  int tsize = state.range(0);
  int upperVal = state.range(1);
  int lRle = state.range(2);
  int uRle = state.range(3);
  int r = state.range(4);
  bool isDense = r == 0;
  Format rv({RLE_s(r)});

  updateCurrent(tsize, upperVal, lRle, uRle, isDense);
  auto res = getCurrentR(tsize, upperVal, lRle, uRle, r);

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