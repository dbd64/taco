#include "../test/test.h"
#include "taco/tensor.h"

#include <benchmark/benchmark.h>
#include <random>
#include <iterator>

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

void IterBounds_help(std::vector<int> dims, std::vector<int> prefix, coord_vector& result){
  if(dims.empty()) return;
  if (dims.size() == 1) {
    for (int i_=0; i_<dims.front(); i_++ ){
      std::vector<int> v = prefix;
      v.push_back(i_);
      result.push_back(v);
    }
  } else {
    auto front = dims.front();
    dims.erase(dims.begin());

    for (int i_ = 0; i_ < front; i_++) {
      std::vector<int> v = prefix;
      v.push_back(i_);
      IterBounds_help(dims, v, result);
    }
  }
}


coord_vector IterBounds(std::vector<int> dims){
  coord_vector result;
  std::vector<int> prefix;
  IterBounds_help(dims, prefix, result);
  return result;
}

template <typename T=double>
Tensor<T> gen_random_data(std::string name, std::vector<int> dims,
                          int lower_rle=1, int upper_rle=512,
                          int lower=0, int upper=1) {
  std::default_random_engine gen;
  std::uniform_int_distribution<int> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<T> unif_vals(lower, upper);

  if (!name.empty()) { name = "_" + name; }

  vector<ModeFormatPack> modes;
  for (auto& _ : dims){
    modes.push_back(Dense);
  }
  const Format dtns(modes);

  Tensor<double> r("d" + name, {dims}, dtns);
  if(dims.size() == 1){
    int index = 0;
    while (index < dims[0]){
      int numCopies = min(unif_rle(gen), dims[0]-index);
      T val = unif_vals(gen);
      for (int i=0; i< numCopies; i++){
        r.insert({index+i}, val);
      }
      index+=numCopies;
    }
  } else {
    auto dims_copy = dims;
    dims_copy.pop_back();
    for(auto& coord_prefix: IterBounds(dims_copy)){
      std::cout << coord_prefix << std::endl;
      int index = 0;
      while (index < dims[0]){
        int numCopies = min(unif_rle(gen), dims[0]-index);
        T val = unif_vals(gen);
        for (int i_=0; i_< numCopies; i_++){
          auto coord = coord_prefix;
          coord.push_back(index+i_);
          r.insert(coord, val);
        }
        index+=numCopies;
      }
    }
  }
  r.pack();

//  Tensor<double> d("d"+name, dims, dv);
//  d(i) = r(i);
//  d.evaluate();

  return r;
}

std::vector<std::vector<int>> getSizes(){
  std::vector<std::vector<int>> result;

  // Generate vector sizes
  for (int s=100; s<=1'000'000'000; s*=10){
    result.push_back({s});
  }

  // Generate square matrix sizes
  for (int s=100; s<=100'000; s*=100){
    result.push_back({s,s});
  }

  // Generate rectangular matrix sizes
  for (int s0=100; s0<=100'000; s0*=100){
    for (int s1=100; s1<=100'000; s1*=100) {
      if(s0!=s1) { result.push_back({s0, s1}); }
    }
  }

  // Generate some higher order tensors
  result.push_back({100,100,1000});
  result.push_back({3,1000,2000,6});

  return result;
}


const std::vector<std::vector<int>> sizes = getSizes();
const std::vector<std::pair<int,int>> rle_ranges =
        {{1,10}, {1,512}, {1,1024}, {512,1024},
         {1000,10000}, {10000,100000}};
const std::vector<std::pair<int,int>> val_ranges =
        {{0,1}, {0,100}};
constexpr int numRandTensors = 1;

template<class T>
std::string name(std::string prefix, std::vector<int> size, int rl, int ru, T vl, T vu, int number){
  std::stringstream r;
  r << "rand_" << prefix << "_s_";
  for(auto s : size){
    r << s << "_";
  }

  r << "r_" << rl << "_" << ru << "_";
  r << "v_" << vl << "_" << vu << "_";

  r << "n_" << number;
  r<< ".tns";

  return r.str();
}

int main(){
  std::string data_out = "/Users/danieldonenfeld/Developer/taco/rle_test/data/";

  for(auto& size : sizes){
    for (auto& [rl,ru] : rle_ranges){
      for (auto& [vl, vu] : val_ranges) {
        for (int k = 0; k < numRandTensors; ++k) {
          auto d0= gen_random_data<double>("0", size, rl, ru, vl, vu);
          auto name0 = name<double>("0", size, rl, ru, vl, vu, k);
          write(data_out + name0, d0);

          std::cout << name0 << std::endl;

          auto d1= gen_random_data<double>("1", size, rl, ru, vl, vu);
          auto name1 = name<double>("1", size, rl, ru, vl, vu, k);
          write(data_out + name1, d1);
        }
      }
    }
  }

}