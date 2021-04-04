//     taco::util::Timer timer;                         \
//     for(int i=0; i<REPEAT; i++) {                    \
//       if(COLD)                                       \
//         timer.clear_cache();                         \
//       timer.start();                                 \
//       CODE;                                          \
//       timer.stop();                                  \
//     }                                                \
//     RES = timer.getResult();                         \
//   }


#include "vec_rle.h"
#include "vec_dense.h"
#include "taco/tensor.h"
#include "taco/util/timers.h"

#include <random>
#include <cstdlib>
#include <cstdint>
#include <benchmark/benchmark.h>

using namespace taco;

//Format rv({RLE_s(16)});
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
gen_random_rle(std::string name, int bits = 16, int size = 100, int lower_rle = 1,
               int upper_rle = 512, int lower = 0, int upper = 1) {
  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<T> unif_vals(lower, upper);

  if (name.empty()) {
    name = "_";
  } else {
    name = "_" + name + "_";
  }

  Format rv({RLE_s(bits)});
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

template <class T, class R>
void fill_rle(Tensor<T> &tt, int size = 100,
              int lower_rle = 1, int upper_rle = 512,
              int lower = 0, int upper = 1) {
  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<int> unif_vals(lower, upper);
  uint64_t max_rle = std::numeric_limits<R>::max();

  taco_tensor_t *t = tt.getStorage();

  // New arrays
  int32_t *t_pos = (int32_t *)malloc(sizeof(int32_t) * 2);
  T *t_vals = (T *)malloc(sizeof(T) * size);
  R *t_rle = (R *)malloc(sizeof(R) * size);

  //Iterator vars
  int index = 0;
  int tnpos = 0;

  while (index < size) {
    uint64_t numCopies = std::min(unif_rle(gen), size - index);
    T val = (T) unif_vals(gen);

    for (uint64_t i = 0; i < (numCopies/max_rle); i++) {
      t_vals[tnpos] = val;
      t_rle[tnpos] = max_rle;
      tnpos++;
    }
    if(numCopies %max_rle != 0 ){
      t_vals[tnpos] = val;
      t_rle[tnpos] = numCopies % max_rle;
      tnpos++;
    }

    index += numCopies;
  }

  t_pos[0] = 0;
  t_pos[1] = tnpos;

  t->indices[0][0] = (uint8_t *)t_pos;

  void *res = realloc(t_vals, sizeof(T) * (tnpos));
  if (!res) {
    taco_uerror;
  }
  t->vals = (uint8_t *)res;

  res = realloc(t_rle, sizeof(R) * (tnpos));
  if (!res) {
    taco_uerror;
  }
  t->indices[0][1] = (uint8_t *)res;


  tt.content->valuesSize = unpackTensorData(*t, tt);
}

template <class T> void fill_rle_b(Tensor<T> &tt, int bits = 16, int size = 100,
                                       int lower_rle = 1, int upper_rle = 512,
                                       int lower = 0, int upper = 1) {
  switch (bits) {
    case 8:
      fill_rle<T, uint8_t>(tt, size, lower_rle, upper_rle, lower, upper);
      break;
    case 16:
      fill_rle<T, uint16_t>(tt, size, lower_rle, upper_rle, lower, upper);
      break;
    case 32:
      fill_rle<T, uint32_t>(tt, size, lower_rle, upper_rle, lower, upper);
      break;
    case 64:
      fill_rle<T, uint64_t>(tt, size, lower_rle, upper_rle, lower, upper);
      break;
    default:
      taco_uerror;
  }
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

template<class T>
void print_rle_b(taco_tensor_t* t, int bits, std::string name) {
  switch (bits) {
    case 8:
      print_rle<T, uint8_t>(t, name);
      break;
    case 16:
      print_rle<T, uint16_t>(t, name);
      break;
    case 32:
      print_rle<T, uint32_t>(t, name);
      break;
    case 64:
      print_rle<T, uint64_t>(t, name);
      break;
    default:
      taco_uerror;
  }}

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

#define TIME_REPEAT(CODE, REPEAT, TIMER, RES, COLD) {  \
    for(int i=0; i<REPEAT; i++) {                    \
      if(COLD)                                       \
        timer.clear_cache();                         \
      TIMER.start();                                 \
      CODE;                                          \
      TIMER.stop();                                  \
    }                                                \
    RES = TIMER.getResult();                         \
  }


void rle_bench(int size, int bits, int repeat, Tensor<double>& r0, Tensor<double>& r1){
  taco::util::Timer timer;
  taco::util::TimeResults timeRLE{};
  {
    Format rv({RLE_s(bits)});
    Tensor<double> expected("expected_", {size}, rv);
    taco_tensor_t* l = r0.getStorage();
    taco_tensor_t* r = r1.getStorage();

    TIME_REPEAT({
       auto e = init_expected(size, false);
       compute(e,l,r);
       benchmark::DoNotOptimize(expected);
       benchmark::ClobberMemory();
       deinit_taco_tensor_t(e);
     }, repeat, timer, timeRLE, false);
  }
  for (auto& time : timer.times){
    std::cout << time << ",";
  }
  std::cout << std::endl;

//  std::cout << "RLE:" << std::endl
//            << timeRLE << std::endl;
}

void dense_bench(int size, int repeat, Tensor<double>& d0, Tensor<double>& d1){
  taco::util::Timer timer;
  taco::util::TimeResults timeDense{};
  {
    Tensor<double> expected("expected_", {size}, dv);
    auto l = d0.getStorage();
    auto r = d1.getStorage();
    TIME_REPEAT({
                       auto e = init_expected(size, true);
                       compute_dense(e,l,r);
                       benchmark::DoNotOptimize(expected);
                       benchmark::ClobberMemory();
                       deinit_taco_tensor_t(e);
                     }, repeat, timer, timeDense, false);
  }
  for (auto& time : timer.times){
    std::cout << time << ",";
  }
  std::cout << std::endl;
//  std::cout << "Dense:" << std::endl
//            << timeDense << std::endl;
}

#ifndef SIZE
#define SIZE 10'000
#endif

#ifndef REPEAT
#define REPEAT 1'000
#endif

#ifndef LOWER_RLE
#define LOWER_RLE 1'000
#endif

#ifndef UPPER_RLE
#define UPPER_RLE 10'000
#endif

#ifndef UPPER_VAL
#define UPPER_VAL 1
#endif

#ifndef RLE_BITS
#define RLE_BITS 16
#endif


int main(int argc, char** argv) {
  int size = SIZE;
  int repeat = REPEAT;
  int lower_rle = LOWER_RLE;
  int upper_rle = UPPER_RLE;
  int upper_val = UPPER_VAL;
  int bits = RLE_BITS;

//  Format rv({RLE_s(bits)});
//  Tensor<double> r0("r0", {size}, rv);
//  Tensor<double> r1("r1", {size}, rv);
//  fill_rle_b<double>(r0, bits, size, lower_rle, upper_rle, 0, upper_val);
//  fill_rle_b<double>(r1, bits, size, lower_rle, upper_rle, 0, upper_val);

  auto r0 = gen_random_rle<double>("0", bits, size, lower_rle, upper_rle, 0, upper_val);
  auto r1 = gen_random_rle<double>("1", bits, size, lower_rle, upper_rle, 0, upper_val);
  compress_rle_b<double>(r0, bits);
  compress_rle_b<double>(r1, bits);

//  std::cout << r0 << std::endl << std::endl;
//  std::cout << r1 << std::endl << std::endl;
//
//  print_rle_b<double>(r0.getTacoTensorT(), bits, "r0");
//  print_rle_b<double>(r1.getTacoTensorT(), bits, "r1");

  rle_bench(size, bits, repeat, r0, r1);

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
