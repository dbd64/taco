#include "taco/tensor.h"
#include "test.h"
#include "test_tensors.h"

#include <random>

using namespace taco;

const Format dv({Dense});
const Format rv({RLE});
const Format dm({Dense, Dense});
const Format rm({Dense, RLE});
const IndexVar i("i"), j("j"), k("k");

std::default_random_engine gen(0);

template <typename T = double, int size = 100>
Tensor<T> gen_random_dense(std::string name, int lower = 0, int upper = 1) {
  std::uniform_int_distribution<T> unif(lower, upper);

  if (name.empty()) {
    name = "_";
  } else {
    name = "_" + name + "_";
  }

  Tensor<T> d("d" + name, {size}, dv);
  for (int i = 0; i < d.getDimension(0); ++i) {
    d.insert({i}, unif(gen));
  }
  d.pack();

  return d;
}

template <typename T = double, int size = 100>
std::pair<Tensor<T>, Tensor<T>> gen_random(std::string name = "", int lower = 0,
                                           int upper = 1) {
  Tensor<T> d = gen_random_dense<T, size>(name, lower, upper);

  if (name.empty()) {
    name = "_";
  } else {
    name = "_" + name + "_";
  }

  Tensor<T> r("r" + name, {size}, rv);
  r(i) = d(i);
  r.setAssembleWhileCompute(true);
  r.compile();
  r.compute();

  return {r, d};
}

template <typename T = double, int size = 100>
std::pair<Tensor<T>, Tensor<T>>
gen_random_rle(std::string name, int lower_rle = 1, int upper_rle = 10,
               int lower = 0, int upper = 1) {
  std::uniform_int_distribution<T> unif_rle(lower_rle, upper_rle);
  std::uniform_int_distribution<T> unif_vals(lower, upper);

  if (name.empty()) {
    name = "_";
  } else {
    name = "_" + name + "_";
  }

  Tensor<T> r("r" + name, {size}, rv);
  int index;
  while (index < size) {
    int numCopies = min(unif_rle(gen), size - index);
    int val = unif_vals(gen);
    for (int i = 0; i < numCopies; i++) {
      r.insert({index + i}, val);
    }
    index += numCopies;
  }
  r.pack();

  Tensor<double> d("d" + name, {size}, dv);
  d(i) = r(i);
  d.evaluate();

  return {r, d};
}

TEST(rlemode, dense_roundtrip) {
  auto [r, d] = gen_random();
  Tensor<double> res("res_", {100}, dv);
  res(i) = r(i);
  res.evaluate();

  ASSERT_TENSOR_EQ(d, res);
}

#define SCOPED_TRACE_NUM(message, num)                                         \
  ::testing::internal::ScopedTrace GTEST_CONCAT_TOKEN_(gtest_trace_, num)(     \
      __FILE__, num, ::testing::Message() << (message))

template <typename type, int size>
void rle_test(IndexExpr (*f)(IndexExpr, IndexExpr)) {
  auto [r0, d0] = gen_random<type, size>("0");
  auto [r1, d1] = gen_random<type, size>("1");
  Tensor<type> expected("expected_", {size}, dv);
  Tensor<type> rle_res("rle_res_", {size}, dv);
  expected(i) = f(d0(i), d1(i));
  rle_res(i) = f(r0(i), r1(i));
  expected.evaluate();
  rle_res.evaluate();
  ASSERT_TENSOR_EQ(expected, rle_res);
}

template <typename type, int size>
void rle_test_into_rle(IndexExpr (*f)(IndexExpr, IndexExpr)) {
  auto [r0, d0] = gen_random<type, size>("0");
  auto [r1, d1] = gen_random<type, size>("1");
  Tensor<type> expected("expected", {size}, dv);
  Tensor<type> rle_res("rle_res", {size}, rv);
  Tensor<type> dense_res("dense_res", {size}, dv);
  expected(i) = f(d0(i), d1(i));
  expected.evaluate();

  rle_res(i) = f(r0(i), r1(i));
  rle_res.setAssembleWhileCompute(true);
  rle_res.compile();
  rle_res.compute();

  dense_res(i) = rle_res(i);
  dense_res.evaluate();

  SCOPED_TRACE_NUM(string("d0: ") + util::toString(d0), 10023);
  SCOPED_TRACE_NUM(string("d1: ") + util::toString(d1), 10024);
  SCOPED_TRACE_NUM(string("rle_res: ") + util::toString(rle_res), 10027);
  ASSERT_TENSOR_EQ(expected, dense_res);
}

#define RLE_TEST_VEC(name, op, type, size)                                     \
  TEST(rlemode, vec_rle_##name##_##type) {                                     \
    rle_test_into_rle<type, size>(                                             \
        [](IndexExpr v1, IndexExpr v2) { return v1 op v2; });                  \
  }                                                                            \
  TEST(rlemode, vec_##name##_##type) {                                         \
    rle_test<type, size>([](IndexExpr v1, IndexExpr v2) { return v1 op v2; }); \
  }

RLE_TEST_VEC(sum_rle, +, double, 100)
RLE_TEST_VEC(sum_rle_large, +, double, 1000)
RLE_TEST_VEC(sum_rle, +, int, 100)
RLE_TEST_VEC(sum_rle_large, +, int, 1000)

RLE_TEST_VEC(sub_rle, -, double, 100)
RLE_TEST_VEC(sub_rle_large, -, double, 1000)

RLE_TEST_VEC(mul_rle, *, double, 100)
RLE_TEST_VEC(mul_rle_large, *, double, 1000)

RLE_TEST_VEC(div_rle, /, double, 100)
RLE_TEST_VEC(div_rle_large, /, double, 1000)

#define RLE_TEST_VEC_2_OP(name, op1, op2, type, size)                          \
  TEST(rlemode, vec_2_op_##name##_##type) {                                    \
    auto [r0, d0] = gen_random<type, size>("0");                               \
    auto [r1, d1] = gen_random<type, size>("1");                               \
    auto [r2, d2] = gen_random<type, size>("2");                               \
    Tensor<type> expected("expected_", {size}, dv);                            \
    Tensor<type> rle_res("rle_res_", {size}, dv);                              \
    expected(i) = d0(i) op1 d1(i) op2 d2(i);                                   \
    rle_res(i) = r0(i) op1 r1(i) op2 r2(i);                                    \
    expected.evaluate();                                                       \
    rle_res.evaluate();                                                        \
    ASSERT_TENSOR_EQ(expected, rle_res);                                       \
  }                                                                            \
  TEST(rlemode, vec_2_op_rle_##name) {                                         \
    auto [r0, d0] = gen_random<type, size>("0");                               \
    auto [r1, d1] = gen_random<type, size>("1");                               \
    auto [r2, d2] = gen_random<type, size>("2");                               \
    Tensor<type> expected("expected_", {size}, dv);                            \
    Tensor<type> rle_res_rle("rle_res_rle", {size}, rv);                       \
    Tensor<type> rle_res("rle_res_", {size}, dv);                              \
    rle_res_rle.setAssembleWhileCompute(true);                                 \
    expected(i) = d0(i) op1 d1(i) op2 d2(i);                                   \
    rle_res_rle(i) = r0(i) op1 r1(i) op2 r2(i);                                \
    rle_res(i) = rle_res_rle(i);                                               \
    expected.evaluate();                                                       \
    rle_res_rle.compile();                                                     \
    rle_res_rle.compute();                                                     \
    rle_res.evaluate();                                                        \
    ASSERT_TENSOR_EQ(expected, rle_res);                                       \
  }

RLE_TEST_VEC_2_OP(sum_rle, +, +, double, 100)
RLE_TEST_VEC_2_OP(mul_rle, *, *, double, 100)
RLE_TEST_VEC_2_OP(sub_rle, -, -, double, 100)
RLE_TEST_VEC_2_OP(div_rle, /, /, double, 100)

RLE_TEST_VEC_2_OP(sum_mul_rle, +, *, double, 100)
RLE_TEST_VEC_2_OP(mul_sum_rle, *, +, double, 100)

RLE_TEST_VEC_2_OP(sum_sub_rle, +, -, double, 100)
RLE_TEST_VEC_2_OP(sub_sum_rle, -, +, double, 100)

RLE_TEST_VEC_2_OP(sum_div_rle, +, /, double, 100)
RLE_TEST_VEC_2_OP(div_sum_rle, /, +, double, 100)

RLE_TEST_VEC_2_OP(mul_sub_rle, *, -, double, 100)
RLE_TEST_VEC_2_OP(sub_mul_rle, -, *, double, 100)

RLE_TEST_VEC_2_OP(mul_div_rle, *, /, double, 100)
RLE_TEST_VEC_2_OP(div_mul_rle, /, *, double, 100)

template <typename T = double>
Tensor<T> gen_random_dense_mat(std::string name, int sizer = 100,
                               int sizec = 100, int lower = 0, int upper = 1) {
  std::uniform_int_distribution<T> unif(lower, upper);

  if (name.empty()) {
    name = "_";
  } else {
    name = "_" + name + "_";
  }

  Tensor<double> d("d" + name, {sizer, sizec}, dm);
  for (int i = 0; i < d.getDimension(0); ++i) {
    for (int j = 0; j < d.getDimension(1); ++j) {
      d.insert({i, j}, unif(gen));
    }
  }
  d.pack();

  return d;
}

template <typename T = double>
std::pair<Tensor<T>, Tensor<T>> gen_random_mat(std::string name = "",
                                               int sizer = 100, int sizec = 100,
                                               int lower = 0, int upper = 1) {
  Tensor<T> d = gen_random_dense_mat<T>(name, sizer, sizec, lower, upper);

  if (name.empty()) {
    name = "_";
  } else {
    name = "_" + name + "_";
  }

  Tensor<double> r("r" + name, {sizer, sizec}, rm);
  r(i, j) = d(i, j);
  r.setAssembleWhileCompute(true);
  r.compile();
  r.compute();

  return {r, d};
}

#define RLE_TEST_MAT(name, op, type, sizer, sizec)                             \
  TEST(rlemode, mat_##name) {                                                  \
    auto [r0, d0] = gen_random_mat<type>("0", sizer, sizec);                   \
    auto [r1, d1] = gen_random_mat<type>("1", sizer, sizec);                   \
    Tensor<type> expected("expected_", {sizer, sizec}, dm);                    \
    Tensor<type> rle_res("rle_res_", {sizer, sizec}, dm);                      \
    expected(i, j) = d0(i, j) op d1(i, j);                                     \
    rle_res(i, j) = r0(i, j) op r1(i, j);                                      \
    expected.evaluate();                                                       \
    rle_res.evaluate();                                                        \
    ASSERT_TENSOR_EQ(expected, rle_res);                                       \
  }

RLE_TEST_MAT(sum_rle_sq, +, double, 100, 100)
RLE_TEST_MAT(sum_rle_sq_large, +, double, 1000, 1000)
RLE_TEST_MAT(sum_rle_rect0, +, double, 100, 200)
RLE_TEST_MAT(sum_rle_rect1, +, double, 200, 100)

RLE_TEST_MAT(sub_rle_sq, -, double, 100, 100)
RLE_TEST_MAT(sub_rle_sq_large, -, double, 1000, 1000)
RLE_TEST_MAT(sub_rle_rect0, -, double, 100, 200)
RLE_TEST_MAT(sub_rle_rect1, -, double, 200, 100)

RLE_TEST_MAT(mul_rle_sq, *, double, 100, 100)
RLE_TEST_MAT(mul_rle_sq_large, *, double, 1000, 1000)
RLE_TEST_MAT(mul_rle_rect0, *, double, 100, 200)
RLE_TEST_MAT(mul_rle_rect1, *, double, 200, 100)

RLE_TEST_MAT(div_rle_sq, /, double, 100, 100)
RLE_TEST_MAT(div_rle_sq_large, /, double, 1000, 1000)
RLE_TEST_MAT(div_rle_rect0, /, double, 100, 200)
RLE_TEST_MAT(div_rle_rect1, /, double, 200, 100)

// TEST(rlemode, tns3) {
//  const Format dt({Dense, Dense, Dense});
//  const Format rt({Dense, Dense, RLE});
//  int lower = 0;
//  int upper = 1;
//
//  std::uniform_int_distribution<double> unif(lower, upper);
//
//  Tensor<double> d0("d0", {100, 100, 100}, dt);
//  Tensor<double> d1("d1", {100, 100, 100}, dt);
//  for (int i = 0; i < d0.getDimension(0); ++i) {
//    for (int j = 0; j < d0.getDimension(1); ++j) {
//      for (int k = 0; j < d0.getDimension(1); ++k) {
//        d0.insert({i, j, k}, unif(gen));
//        d1.insert({i, j, k}, unif(gen));
//      }
//    }
//  }
//  d0.pack();
//  d1.pack();
//
//  Tensor<double> r0("r0", {100, 100, 100}, rt);
//  r0(i, j, k) = d0(i, j, k);
//  r0.setAssembleWhileCompute(true);
//  r0.compile();
//  r0.compute();
//  Tensor<double> r1("r1", {100, 100, 100}, rt);
//  r1(i, j, k) = d1(i, j, k);
//  r1.setAssembleWhileCompute(true);
//  r1.compile();
//  r1.compute();
//
//  Tensor<double> expected("expected_", {100, 100, 100}, dt);
//  Tensor<double> rle_res("rle_res_", {100, 100, 100}, dt);
//
//  expected(i, j, k) = d0(i, j, k) + d1(i, j, k);
//  rle_res(i, j, k) = r0(i, j, k) + r1(i, j, k);
//  expected.evaluate();
//  rle_res.evaluate();
//  ASSERT_TENSOR_EQ(expected, rle_res);
//}
