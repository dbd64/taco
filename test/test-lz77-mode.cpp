#include "taco/tensor.h"
#include "test.h"
#include "taco.h"
#include "../src/lower/iteration_graph.h"

#include <limits>
#include <random>

using namespace taco;

const Format dv({Dense});
const Format lz77f({LZ77});

const IndexVar i("i");

Index makeLZ77Index(const std::vector<int>& rowptr, const std::vector<int>& dist,
                    const std::vector<int>& runs) {
  return Index(lz77f,
               {ModeIndex({makeArray(rowptr), makeArray(dist), makeArray(runs)})});
}

template<typename T>
TensorBase makeLZ77(const std::string& name, const std::vector<int>& dimensions,
                    const std::vector<int>& pos,
                    const std::vector<int>& dist,
                    const std::vector<int>& runs,
                    const std::vector<T>& vals) {
  taco_uassert(dimensions.size() == 1);
  Tensor<T> tensor(name, dimensions, {LZ77});
  auto storage = tensor.getStorage();
  storage.setIndex(makeLZ77Index(pos, dist, runs));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

Tensor<double> lz77_zeros(std::string name) {
  return makeLZ77<double>(name, {11},
                          {0, 11},
                          {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0  },
                          {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0  },
                          {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0});
}

Tensor<double> lz77_one_rle(std::string name, double val) {
  return makeLZ77<double>(name, {11},
                          {0, 2},
                          {1,  0  },
                          {9,  0  },
                          {val,0.0});
}

Tensor<double> lz77_two_repeat(std::string name, double val1, double val2) {
  return makeLZ77<double>(name, {11},
                          {0, 3},
                          {0,   2,   0  },
                          {0,   8,   0  },
                          {val1,val2,0.0});
}

Tensor<double> lz77_repeat_twice(std::string name, double val1, double val2) {
  return makeLZ77<double>(name, {11},
                          {0, 3},
                          {1,   2,   0  },
                          {4,   4,   0  },
                          {val1,val2,0.0});
}

Tensor<double> lz77_three_repeat(std::string name, double val1, double val2, double val3) {
  return makeLZ77<double>(name, {11},
                          {0, 4},
                          {0,   0,   3,  0   },
                          {0,   0,   7,  0   },
                          {val1,val2,val3,0.0});
}

Tensor<double> lz77_1(std::string name) {
  return makeLZ77<double>(name, {11},
                          {0, 7},
                          {0,  0,  1,  0,  0,  3,  0  },
                          {0,  0,  3,  0,  0,  1,  0  },
                          {0.0,1.0,2.0,3.0,4.0,5.0,0.0});
}

Tensor<double> lz77_2(std::string name) {
  return makeLZ77<double>(name, {11},
                          {0, 7},
                          {0,  0,  1,  0,  0,  3,  0  },
                          {0,  0,  2,  0,  0,  2,  0  },
                          {0.0,1.0,2.0,3.0,4.0,5.0,0.0});
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
