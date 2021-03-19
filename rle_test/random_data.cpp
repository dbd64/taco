#include "../test/test.h"
#include "taco/tensor.h"

#include <benchmark/benchmark.h>
#include <random>
#include <iterator>

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v){
  if (v.empty()) { out << "[]"; }

  out << "[";
  for(unsigned long i=0; i<v.size()-1; i++){
    out << v[i] << ", ";
  }
  out << v[v.size()-1] << "]";

  return out;
}



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

using coord_vector = std::vector<std::vector<int>>;

coord_vector IterBounds_tail(std::vector<int> dims, std::vector<int> prefix, coord_vector& result){
  if(dims.empty()) return coord_vector(0);
  if (dims.size() == 1) {
    coord_vector result;
    for (int i_=0; i_<dims.front(); i_++ ){
      std::vector<int> v = {i_};
      result.push_back(v);
    }
    std::cout << "base case : " << result << std::endl;
    return result;
  }

  prefix.push_back(dims.front());
  dims.erase(dims.begin());

}


coord_vector IterBounds(std::vector<int> dims){
  if(dims.empty()) return coord_vector(0);
  if (dims.size() == 1) {
    coord_vector result;
    for (int i_=0; i_<dims.front(); i_++ ){
      std::vector<int> v = {i_};
      result.push_back(v);
    }
    std::cout << "base case : " << result << std::endl;
    return result;
  }

  int dim_n = dims.back();
  dims.pop_back();
  coord_vector recurse = IterBounds(dims);

  coord_vector result;
  for (int i_ = 0; i_ < dim_n; ++i_) {
    for(auto e : recurse){
      e.push_back(i_);
      result.push_back(e);
    }
  }

  std::cout << "recursive case : " << result << std::endl;
  return result;
}
//
//template <typename T=double>
//Tensor<T> gen_random_data(std::string name, std::vector<int> dims,
//                          int lower_rle=1, int upper_rle=512,
//                          int lower=0, int upper=1) {
//  std::default_random_engine gen;
//  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
//  std::uniform_int_distribution<T> unif_vals(lower, upper);
//
//  if (!name.empty()) { name = "_" + name; }
//
//  Tensor<double> r("r" + name, {dims}, rv);
//  int index = 0;
//  while (index < std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>())){
//    int numCopies = min(unif_rle(gen), size-index);
//    T val = unif_vals(gen);
//    for (int i=0; i< numCopies; i++){
//      r.insert({index+i}, val);
//    }
//    index+=numCopies;
//  }
//  r.pack();
//
//  Tensor<double> d("d"+name, {size}, dv);
//  d(i) = r(i);
//  d.evaluate();
//
//  return {r,d};
//}

std::vector<std::pair<int,int>> rle_ranges =
        {{1,512}, {1,1024}, {512,1024}, {1000,10000},
         {10000,100000}, {1, 10}};
constexpr int size_lower = 100;
constexpr int size_upper = 10'000'000;
constexpr int size_mult = 10;
constexpr int val_lower = 1;
constexpr int val_upper = 100;
constexpr int val_mult = 10;

constexpr int numRandTensors = 50;


int main(){
//  std::vector<int> dims = {2,2,3};
  std::cout << IterBounds({2,2,3}) << std::endl;

//  for(int i=size_lower; i<=size_upper; i*=size_mult) { // Size of vector
//    for (int j = val_lower; j <= val_upper; j*=val_mult) { // Upper limit of values
//      for (auto &x : rle_ranges) { // bounds on rle values
//        auto [l, u] = x;
//        for (int k=0; k< numRandTensors; k++) {
//          std::cout << "loop : " << i <<", " << j << ", " << l << ", " << u << std::endl;
//
//          auto[r0, d0] = gen_random_rle<double>("0", i, l, u, 0, j);
//          auto[r1, d1] = gen_random_rle<double>("1", i, l, u, 0, j);
//        }
//      }
//    }
//  }
}