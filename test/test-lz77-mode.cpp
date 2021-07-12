#include "taco/tensor.h"
#include "test.h"
#include "taco.h"
#include "../src/lower/iteration_graph.h"

#include <limits>
#include <random>
#include <variant>

using namespace taco;

const Format dv({Dense});
const Format lz77f({LZ77});

const IndexVar i("i");

template <typename T>
union GetBytes {
    T value;
    uint8_t bytes[sizeof(T)];
};

using Repeat = std::pair<uint16_t, uint16_t>;

template <class T>
using TempValue = std::variant<T,Repeat>;

// helper type for the visitor #4
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename T>
void push_back(T arg, std::vector<uint8_t>& bytes, bool check = false){
  GetBytes<T> gb;
  gb.value = arg;

  if(check && ((gb.bytes[0] >> 7) & 0x1) == 1){
    bytes.push_back(1 << 7);
  }

  for (unsigned long i_=0; i_<sizeof(T); i_++){
    bytes.push_back(gb.bytes[i_]);
  }
}

template <typename T>
std::pair<std::vector<uint8_t>, int> packLZ77(std::vector<TempValue<T>> vals){
  std::vector<uint8_t> bytes;
  for (auto& val : vals){
    std::visit(overloaded {
            [&](T arg) { push_back(arg, bytes, true); },
            [&](std::pair<uint16_t, uint16_t> arg) {
                bytes.push_back(0x3 << 6);
                push_back(arg.first, bytes); push_back(arg.second, bytes);
            }
    }, val);
  }

  return {bytes, bytes.size()};
//  int size = bytes.size();
//  while(bytes.size() % sizeof(T) != 0){
//    bytes.push_back(0);
//  }
//  T* bytes_data = (T*) bytes.data();
//  std::vector<T> values(bytes_data, bytes_data + (bytes.size() / sizeof(T)));
//
//  return {values, size};
}

Index makeLZ77Index(const std::vector<int>& rowptr) {
  return Index(lz77f,
               {ModeIndex({makeArray(rowptr)})});
}

/// Factory function to construct a compressed sparse row (CSR) matrix.
template<typename T>
TensorBase makeLZ77(const std::string& name, const std::vector<int>& dimensions,
                    const std::vector<int>& pos, const std::vector<uint8_t>& vals) {
  taco_uassert(dimensions.size() == 1);
  Tensor<T> tensor(name, dimensions, {LZ77});
  auto storage = tensor.getStorage();
  storage.setIndex(makeLZ77Index(pos));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

Tensor<double> lz77_zeros(std::string name) {
  auto packed = packLZ77<double>({0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0});
  return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
}

Tensor<double> lz77_one_rle(std::string name, double val) {
  auto packed = packLZ77<double>({val,Repeat{1,9},0.0});
  return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
}


Tensor<double> lz77_two_repeat(std::string name, double val1, double val2) {
  auto packed = packLZ77<double>({val1,val2,Repeat{2,8},0.0});
  return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
}

Tensor<double> lz77_repeat_twice(std::string name, double val1, double val2) {
  auto packed = packLZ77<double>({val1,Repeat{1,3},val1,val2,Repeat{2,4},0.0});
  return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
}

Tensor<double> lz77_three_repeat(std::string name, double val1, double val2, double val3) {
  auto packed = packLZ77<double>({val1,val2,val3,Repeat{3,7},0.0});
  return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
}

Tensor<double> lz77_1(std::string name) {
  auto packed = packLZ77<double>({0.0,1.0,2.0,Repeat{1,3},3.0,4.0,5.0,Repeat{3,1},0.0});
  return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
}

Tensor<double> lz77_2(std::string name) {
  auto packed = packLZ77<double>({0.0,1.0,2.0,Repeat{1,2},3.0,4.0,5.0,Repeat{3,2},0.0});
  return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
}

//std::default_random_engine gen(0);

//template <typename T>
//std::ostream& operator<<(std::ostream& out, const std::vector<T>& v){
//  if (v.size() == 0) { out << "[]"; return out; }
//
//  out << "[";
//  for(unsigned long i=0; i<v.size()-1; i++){
//    out << v[i] << ", ";
//  }
//  out << v[v.size()-1] << "]";
//
//  return out;
//}
//
//template <typename T>
//constexpr auto rand_dist(T lower, T upper) {
//  if constexpr (is_integral<T>::value) {
//    std::uniform_int_distribution<T> unif_vals(lower, upper);
//    return unif_vals;
//  } else {
//    std::uniform_real_distribution<T> unif_vals(lower, upper);
//    return unif_vals;
//  }
//}
//
//template <typename T = double, int size = 100>
//Tensor<T> gen_random_dense(std::string name = "", int lower = 0, int upper = 1) {
//  auto unif = rand_dist<int>(lower, upper);
//
//  if (name.empty()) {
//    name = "_";
//  } else {
//    name = "_" + name + "_";
//  }
//
//  Tensor<T> d("d" + name, {size}, dv);
//  for (int i = 0; i < d.getDimension(0); ++i) {
//    d.insert({i}, (T) unif(gen));
//  }
//  d.pack();
//
//  return d;
//}
//
//template <typename T = double, int size = 100>
//std::pair<Tensor<T>, Tensor<T>> gen_random(std::string name = "", int bits = 16,
//                                           int lower = 0, int upper = 1) {
//  Tensor<T> d = gen_random_dense<T, size>(name, lower, upper);
//
//  if (name.empty()) {
//    name = "_";
//  } else {
//    name = "_" + name + "_";
//  }
//
//  Tensor<T> r("lz77" + name, {size}, lz77f);
//  r(i) = d(i);
//  r.setAssembleWhileCompute(true);
//  r.compile();
//  r.compute();
//
//  return {r, d};
//}

Func getCopyFunc(){
  auto copyFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], 0);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      return Union(l, Complement(l));
  };
  Func copy("copy_", copyFunc, algFunc);
  return copy;
}

Func getPlusFunc(){
  auto plusFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], v[1]);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      auto r = Region(v[1]);
      return Union(Union(l, r), Union(Complement(l), Complement(r)));
  };
  Func plus_("plus_", plusFunc, algFunc);
  return plus_;
}

TEST(lz77_mode, test_values) {
  Func copy = getCopyFunc();
  Func plus_ = getPlusFunc();

  Tensor<double> A("A", {11}, dv, 0);
  Tensor<double> B = lz77_two_repeat("B", 1,2);

  A(i) = copy(B(i));
  A.setAssembleWhileCompute(true);
  A.compile();
  A.compute();

  Tensor<double> result("A", {11}, dv, 0);
  result(0) = 1;
  result(1) = 2;
  result(2) = 1;
  result(3) = 2;
  result(4) = 1;
  result(5) = 2;
  result(6) = 1;
  result(7) = 2;
  result(8) = 1;
  result(9) = 2;
  result(10) = 0;


  SCOPED_TRACE(string("A: ") + util::toString(A));
  SCOPED_TRACE(string("B: ") + util::toString(B));

  ASSERT_TENSOR_EQ(A, result);
}

TEST(lz77_mode, test_zeros) {
  Func copy = getCopyFunc();
  Func plus_ = getPlusFunc();

  Tensor<double> A("A", {11}, lz77f, 0);
  Tensor<double> B = lz77_zeros("B");
  Tensor<double> C = lz77_zeros("C");
  Tensor<double> result("result", {11}, dv, 0);

  A(i) = plus_(B(i), C(i));
  A.setAssembleWhileCompute(true);
  A.compile();
  A.compute();

  result(i) = copy(A(i));
  result.setAssembleWhileCompute(true);
  result.compile();
  result.compute();

  Tensor<double> A_("dA", {11},   dv, 0);
  Tensor<double> B_("dB", {11},   dv, 0);
  Tensor<double> C_("dC", {11},   dv, 0);

  B_(i) = copy(B(i));
  B_.setAssembleWhileCompute(true);
  B_.compile();
  B_.compute();

  C_(i) = copy(C(i));
  C_.setAssembleWhileCompute(true);
  C_.compile();
  C_.compute();

  A_(i) = B_(i) + C_(i);

  A_.setAssembleWhileCompute(true);
  A_.compile();
  A_.compute();

  SCOPED_TRACE(string("A: ") + util::toString(A));
  SCOPED_TRACE(string("C: ") + util::toString(C));
  SCOPED_TRACE(string("B: ") + util::toString(B));

  SCOPED_TRACE(string("C_: ") + util::toString(C_));
  SCOPED_TRACE(string("B_: ") + util::toString(B_));

  ASSERT_TENSOR_EQ(A_, result);
}

TEST(lz77_mode, test_rle) {
  Func copy = getCopyFunc();
  Func plus_ = getPlusFunc();

  Tensor<double> A("A", {11}, lz77f, 0);
  Tensor<double> B = lz77_one_rle("B", 5);
  Tensor<double> C = lz77_one_rle("C", 7);
  Tensor<double> result("result", {11}, dv, 0);

  A(i) = plus_(B(i), C(i));
  A.setAssembleWhileCompute(true);
  A.compile();
  A.compute();

  result(i) = copy(A(i));
  result.setAssembleWhileCompute(true);
  result.compile();
  result.compute();

  Tensor<double> A_("dA", {11},   dv, 0);
  Tensor<double> B_("dB", {11},   dv, 0);
  Tensor<double> C_("dC", {11},   dv, 0);

  B_(i) = copy(B(i));
  B_.setAssembleWhileCompute(true);
  B_.compile();
  B_.compute();

  C_(i) = copy(C(i));
  C_.setAssembleWhileCompute(true);
  C_.compile();
  C_.compute();

  A_(i) = B_(i) + C_(i);

  A_.setAssembleWhileCompute(true);
  A_.compile();
  A_.compute();

  SCOPED_TRACE(string("A: ") + util::toString(A));
  SCOPED_TRACE(string("C: ") + util::toString(C));
  SCOPED_TRACE(string("B: ") + util::toString(B));

  SCOPED_TRACE(string("C_: ") + util::toString(C_));
  SCOPED_TRACE(string("B_: ") + util::toString(B_));

  ASSERT_TENSOR_EQ(A_, result);
}

TEST(lz77_mode, test_repeat_two) {
  Func copy = getCopyFunc();
  Func plus_ = getPlusFunc();

  Tensor<double> A("A", {11}, lz77f, 0);
  Tensor<double> B = lz77_two_repeat("B", 1,2);
  Tensor<double> C = lz77_two_repeat("C", 3,4);
  Tensor<double> result("result", {11}, dv, 0);

  A(i) = plus_(B(i), C(i));
  A.setAssembleWhileCompute(true);
  A.compile();
  A.compute();

  result(i) = copy(A(i));
  result.setAssembleWhileCompute(true);
  result.compile();
  result.compute();

  Tensor<double> A_("dA", {11},   dv, 0);
  Tensor<double> B_("dB", {11},   dv, 0);
  Tensor<double> C_("dC", {11},   dv, 0);

  B_(i) = copy(B(i));
  B_.setAssembleWhileCompute(true);
  B_.compile();
  B_.compute();

  C_(i) = copy(C(i));
  C_.setAssembleWhileCompute(true);
  C_.compile();
  C_.compute();

  A_(i) = B_(i) + C_(i);

  A_.setAssembleWhileCompute(true);
  A_.compile();
  A_.compute();

  SCOPED_TRACE(string("A: ") + util::toString(A));
  SCOPED_TRACE(string("C: ") + util::toString(C));
  SCOPED_TRACE(string("B: ") + util::toString(B));

  SCOPED_TRACE(string("C_: ") + util::toString(C_));
  SCOPED_TRACE(string("B_: ") + util::toString(B_));

  std::stringstream sstream;
  A.printComputeIR(sstream);
  SCOPED_TRACE(string("Compute code: \n") + sstream.str());

  ASSERT_TENSOR_EQ(A_, result);
}

TEST(lz77_mode, test_mixed_two_three) {
  Func copy = getCopyFunc();
  Func plus_ = getPlusFunc();

  Tensor<double> A("A", {11}, lz77f, 0);
  Tensor<double> B = lz77_two_repeat("B", 1,2);
  Tensor<double> C = lz77_three_repeat("C", 3,4,5);
  Tensor<double> result("result", {11}, dv, 0);

  A(i) = plus_(B(i), C(i));
  A.setAssembleWhileCompute(true);
  A.compile();
  A.compute();

  result(i) = copy(A(i));
  result.setAssembleWhileCompute(true);
  result.compile();
  result.compute();

  Tensor<double> A_("dA", {11},   dv, 0);
  Tensor<double> B_("dB", {11},   dv, 0);
  Tensor<double> C_("dC", {11},   dv, 0);

  B_(i) = copy(B(i));
  B_.setAssembleWhileCompute(true);
  B_.compile();
  B_.compute();

  C_(i) = copy(C(i));
  C_.setAssembleWhileCompute(true);
  C_.compile();
  C_.compute();

  A_(i) = B_(i) + C_(i);

  A_.setAssembleWhileCompute(true);
  A_.compile();
  A_.compute();

  SCOPED_TRACE(string("A: ") + util::toString(A));
  SCOPED_TRACE(string("C: ") + util::toString(C));
  SCOPED_TRACE(string("B: ") + util::toString(B));

  SCOPED_TRACE(string("C_: ") + util::toString(C_));
  SCOPED_TRACE(string("B_: ") + util::toString(B_));

  ASSERT_TENSOR_EQ(A_, result);
}

TEST(lz77_mode, test_mixed) {
  Func copy = getCopyFunc();
  Func plus_ = getPlusFunc();

  Tensor<double> A("A", {11}, lz77f, 0);
  Tensor<double> B = lz77_repeat_twice("B", 1,2);
  Tensor<double> C = lz77_three_repeat("C", 3,4,5);
  Tensor<double> result("result", {11}, dv, 0);

  A(i) = plus_(B(i), C(i));
  A.setAssembleWhileCompute(true);
  A.compile();
  A.compute();

  result(i) = copy(A(i));
  result.setAssembleWhileCompute(true);
  result.compile();
  result.compute();

  Tensor<double> A_("dA", {11},   dv, 0);
  Tensor<double> B_("dB", {11},   dv, 0);
  Tensor<double> C_("dC", {11},   dv, 0);

  B_(i) = copy(B(i));
  B_.setAssembleWhileCompute(true);
  B_.compile();
  B_.compute();

  C_(i) = copy(C(i));
  C_.setAssembleWhileCompute(true);
  C_.compile();
  C_.compute();

  A_(i) = B_(i) + C_(i);

  A_.setAssembleWhileCompute(true);
  A_.compile();
  A_.compute();

  SCOPED_TRACE(string("A: ") + util::toString(A));
  SCOPED_TRACE(string("C: ") + util::toString(C));
  SCOPED_TRACE(string("B: ") + util::toString(B));

  SCOPED_TRACE(string("C_: ") + util::toString(C_));
  SCOPED_TRACE(string("B_: ") + util::toString(B_));

  ASSERT_TENSOR_EQ(A_, result);
}

TEST(lz77_mode, test_examples) {
  Func copy = getCopyFunc();
  Func plus_ = getPlusFunc();

  Tensor<double> A("A", {11}, lz77f, 0);
  Tensor<double> B = lz77_1("B");
  Tensor<double> C = lz77_2("C");
  Tensor<double> result("result", {11}, dv, 0);

  A(i) = plus_(B(i), C(i));
  A.setAssembleWhileCompute(true);
  A.compile();
  A.compute();

  result(i) = copy(A(i));
  result.setAssembleWhileCompute(true);
  result.compile();
  result.compute();

  Tensor<double> A_("dA", {11},   dv, 0);
  Tensor<double> B_("dB", {11},   dv, 0);
  Tensor<double> C_("dC", {11},   dv, 0);

  B_(i) = copy(B(i));
  B_.setAssembleWhileCompute(true);
  B_.compile();
  B_.compute();

  C_(i) = copy(C(i));
  C_.setAssembleWhileCompute(true);
  C_.compile();
  C_.compute();

  A_(i) = B_(i) + C_(i);

  A_.setAssembleWhileCompute(true);
  A_.compile();
  A_.compute();

  SCOPED_TRACE(string("A: ") + util::toString(A));
  SCOPED_TRACE(string("C: ") + util::toString(C));
  SCOPED_TRACE(string("B: ") + util::toString(B));

  SCOPED_TRACE(string("C_: ") + util::toString(C_));
  SCOPED_TRACE(string("B_: ") + util::toString(B_));

  ASSERT_TENSOR_EQ(A_, result);
}

TEST(lz77_mode, test_repeat_two_csr) {
  Func copy = getCopyFunc();
  Func plus_ = getPlusFunc();

  Tensor<double> A("A", {11}, lz77f, 0);
  Tensor<double> B = lz77_two_repeat("B", 1,2);
  Tensor<double> C("C", {11}, {Compressed}, 0);
  Tensor<double> result("result", {11}, dv, 0);

  C(0) = 1;
  C(5) = 2;
  C(9) = 5;
  C(10) = 0;

  A(i) = plus_(B(i), C(i));
  A.setAssembleWhileCompute(true);
  A.compile();
  A.compute();

  result(i) = copy(A(i));
  result.setAssembleWhileCompute(true);
  result.compile();
  result.compute();

  Tensor<double> A_("dA", {11},   dv, 0);
  Tensor<double> B_("dB", {11},   dv, 0);
  Tensor<double> C_("dC", {11},   dv, 0);

  B_(i) = copy(B(i));
  B_.setAssembleWhileCompute(true);
  B_.compile();
  B_.compute();

  C_(i) = copy(C(i));
  C_.setAssembleWhileCompute(true);
  C_.compile();
  C_.compute();

  A_(i) = B_(i) + C_(i);

  A_.setAssembleWhileCompute(true);
  A_.compile();
  A_.compute();

  SCOPED_TRACE(string("A: ") + util::toString(A));
  SCOPED_TRACE(string("C: ") + util::toString(C));
  SCOPED_TRACE(string("B: ") + util::toString(B));

  SCOPED_TRACE(string("C_: ") + util::toString(C_));
  SCOPED_TRACE(string("B_: ") + util::toString(B_));

  ASSERT_TENSOR_EQ(A_, result);
}
