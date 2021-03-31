extern "C" {
  #include "temp_vec.h"
  #include "temp_vec_dense.h"
}
#include "taco/tensor.h"
#include "taco/util/timers.h"

#include <random>
#include <cstdlib>
#include <cstdint>
#include <benchmark/benchmark.h>

using namespace taco;

//Format rv({RLE_s(16)});
Format rv({RLE_s(16)});
Format dv({Dense});

std::default_random_engine gen(0);

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v){
  if (v.size() == 0) { out << "[]"; return out; }

  out << "[";
  for(unsigned long i=0; i<v.size()-1; i++){
    out << v[i] << ", ";
  }
  out << v[v.size()-1] << "]";

  return out;
}

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

  Tensor<double> r("r" + name, {size}, rv);
  int index = 0;
  while (index < size) {
    int numCopies = std::min(unif_rle(gen), size - index);
    T val = unif_vals(gen);
    for (int i = 0; i < numCopies; i++) {
      r.insert({index + i}, val);
    }
    index += numCopies;
  }
  r.pack();
  return r;
}

void printv(taco_tensor_t* t) {
  std::cout << ((double*) t->vals)[0];
  for (int i=0;i<t->dimensions[0]; i++){
    std::cout << ", " << ((double*) t->vals)[i] ;
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

void print(Tensor<double>& t) {
  std::cout << t << std::endl << std::endl;
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
  T* t_vals = (T*)(t->vals);

  T* t_vals_new = (T*)malloc(sizeof(T)*t_pos[1]);
  t_vals_new[0] = t_vals[0];

  int32_t tnpos = 0;
  T tnval = t_vals[0];

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

  void* res = realloc(t_vals_new, sizeof(T) * (tnpos+1));
  if(!res){ taco_uerror; }
  t->vals = (uint8_t*)res;

  res = realloc(t_rle, sizeof(R) * (tnpos+1));
  if(!res){ taco_uerror; }
  t->indices[0][1] = (uint8_t*)res;

  t_pos[1] = tnpos+1;

  tt.content->valuesSize = unpackTensorData(*t, tt);
}

template <class T>
void compress_rle_b(Tensor<T>& tt, int bits){
  switch (bits){
    case 8: compress_rle<T,uint8_t>(tt); break;
    case 16: compress_rle<T,uint16_t>(tt); break;
    case 32: compress_rle<T,uint32_t>(tt); break;
    case 64: compress_rle<T,uint64_t>(tt); break;
    default: taco_uerror;
  }
}

template<class T, class R>
void print_rle(taco_tensor_t* t, std::string name) {
  int32_t* t_pos = (int32_t*)(t->indices[0][0]);
  std::vector<int32_t> pos(t_pos, t_pos+2);

  R* t_rle = (R*)(t->indices[0][1]);
  std::vector<R> rle(t_rle, t_rle+pos[1]);

  T* t_vals = (T*)(t->vals);
  std::vector<T> vals(t_vals, t_vals+pos[1]);

  std::cout << name << ":" << std::endl;
  std::cout << "  pos  : " << pos << std::endl;
  std::cout << "  rle  : " << rle << std::endl;
  std::cout << "  vals : " << vals << std::endl;
}

__attribute__((noinline))
void computePls(Tensor<double> expected, taco_tensor_t *t, taco_tensor_t *f, taco_tensor_t *s){
  compute(t, f, s);
  expected.content->valuesSize = unpackTensorData(*t, expected);
}

__attribute__((noinline))
void computeDense(Tensor<double> expected, taco_tensor_t *t, taco_tensor_t *f, taco_tensor_t *s){
  compute_dense(t, f, s);
  expected.content->valuesSize = unpackTensorData(*t, expected);
}

taco_tensor_t* init_expected(int size, bool isDense){
  int32_t order = 1;
  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
  t->order         = order;
  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
  t->csize         = 64;

  t->dimensions[0]    = size;
  t->mode_ordering[0] = 0;
  t->mode_types[0]    = isDense? taco_mode_dense : taco_mode_sparse;
  switch (t->mode_types[0]) {
    case taco_mode_dense:
      t->indices[0] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
      break;
    case taco_mode_sparse:
      t->indices[0] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
      break;
  }
  return t;
}

void rle_bench(int size, int repeat, Tensor<double>& r0, Tensor<double>& r1){
  taco::util::TimeResults timeRLE{};
  {
    Tensor<double> expected("expected_", {size}, rv);
    taco_tensor_t* l = r0.getStorage();
    taco_tensor_t* r = r1.getStorage();

    TACO_TIME_REPEAT({
       auto e = init_expected(size, false);
       compute(e,l,r);
       benchmark::DoNotOptimize(expected);
       benchmark::ClobberMemory();
       deinit_taco_tensor_t(e);
     }, repeat, timeRLE, false);
  }
  std::cout << "RLE:" << std::endl
            << timeRLE << std::endl;
}

void dense_bench(int size, int repeat, Tensor<double>& d0, Tensor<double>& d1){
  taco::util::TimeResults timeDense{};
  {
    Tensor<double> expected("expected_", {size}, dv);
    auto l = d0.getStorage();
    auto r = d1.getStorage();
    TACO_TIME_REPEAT({
                       auto e = init_expected(size, true);
                       compute_dense(e,l,r);
                       benchmark::DoNotOptimize(expected);
                       benchmark::ClobberMemory();
                       deinit_taco_tensor_t(e);
                     }, repeat, timeDense, false);
  }
  std::cout << "Dense:" << std::endl
            << timeDense << std::endl;
}

int main(int argc, char** argv) {
  int size = 5'000'000;
  int repeat = 1000;

  auto r0 = gen_random_rle<double>("0", size, 1'000, 10'000);
  auto r1 = gen_random_rle<double>("1", size, 1'000, 10'000);

  compress_rle_b<double>(r0,16);
  compress_rle_b<double>(r1, 16);

  std::cout << r0 << std::endl << std::endl;
  std::cout << r1 << std::endl << std::endl;

  std::cout << "STARTING rle!" << std::endl;
  rle_bench(size, repeat, r0, r1);
  std::cout << "DONE rle!" << std::endl;

  {
    IndexVar i("i");
    Tensor<double> d0("d0", {size}, dv);
    d0(i) = r0(i);
    d0.evaluate();

    Tensor<double> d1("d1", {size}, dv);
    d1(i) = r1(i);
    d1.evaluate();
    dense_bench(size, repeat, d0, d1);
  }
}
