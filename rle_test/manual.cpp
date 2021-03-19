extern "C" {
  #include "temp_vec.h"
}
#include "taco/tensor.h"

#include <random>

using namespace taco;

const Format dv({Dense});
const Format rv({RLE});
const IndexVar i("i"), j("j"), k("k");

template <typename T=double, int size=100>
Tensor<T> gen_random_dense(std::string name, int lower=0, int upper=1) {
  std::default_random_engine gen(0);
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
std::pair<Tensor<T>, Tensor<T>> gen_random_rle(std::string name, int size= 100, int lower_rle=3, int upper_rle=6,
                                               int lower=0, int upper=15) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<T> unif_vals(lower, upper);

  if (name.empty()){ name = "_"; }
  else { name = "_" + name + "_"; }

  Tensor<double> r("r" + name, {size}, rv);
  int index = 0;
  while (index < size){
    int numCopies = std::min(unif_rle(gen), size-index);
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
      Array size = makeArray({valsSize}); // TODO: This is wrong
      Array pos = Array(type<int>(), tensorData.indices[i][0], numVals+1, Array::UserOwns);
      Array rle = Array(type<int>(), tensorData.indices[i][1], valsSize, Array::UserOwns);
      modeIndices.push_back(ModeIndex({size, pos, rle}));
      numVals = valsSize;
    } else {
      taco_not_supported_yet;
    }
  }
  storage.setIndex(Index(format, modeIndices));
  storage.setValues(Array(tensor.getComponentType(), tensorData.vals, numVals));
  return numVals;
}

void computePls(Tensor<double> expected, taco_tensor_t *t, taco_tensor_t *f, taco_tensor_t *s){
  compute(t, f, s);
  expected.content->valuesSize = unpackTensorData(*t, expected);
}

int main(int argc, char** argv) {
  auto[r0, d0] = gen_random_rle<double>("0", 10);
  auto[r1, d1] = gen_random_rle<double>("1", 10);

  Tensor<double> expected("expected_", {10}, dv);
  expected.pack();
  print(expected);

  computePls(expected, expected.getStorage(), r0.getStorage(), r1.getStorage());

//  std::cout << r0 << std::endl;
//  std::cout << r1 << std::endl;
  std::cout << d0 << std::endl;
  std::cout << d1 << std::endl;



  print(expected);

//  IndexVar i("i");
//  expected(i) = r0(i) + r1(i);
//  expected.evaluate();
//
//  print(expected);
//  printv(expected.getStorage());

}
