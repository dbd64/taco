#ifndef TACO_VEC_DENSE_H
#define TACO_VEC_DENSE_H
extern "C" {
#include "taco/taco_tensor_t.h"
int compute_dense(taco_tensor_t *t, taco_tensor_t *f, taco_tensor_t *s);
}
#endif //TACO_VEC_DENSE_H
