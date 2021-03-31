//
// Created by Daniel Donenfeld on 3/12/21.
//

#ifndef TACO_TEMP_VEC_H
#define TACO_TEMP_VEC_H

#include "taco/taco_tensor_t.h"

int compute(taco_tensor_t *t, taco_tensor_t *f, taco_tensor_t *s);
void deinit_taco_tensor_t(taco_tensor_t* t);

#endif //TACO_TEMP_VEC_H