// Generated by the Tensor Algebra Compiler (tensor-compiler.org)
// /Users/danieldonenfeld/Developer/taco/cmake-build-release-lanka/bin/taco "t(i) = f(i)+ s(i)" -f=t:r -f=f:r -f=s:r -c -write-source=temp_vec.c
#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_LCM(_a,_b) (TODO)
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
int cmp(const void *a, const void *b) {
  return *((const int*)a) - *((const int*)b);
}
int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types) {
  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
  t->order         = order;
  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
  t->csize         = csize;
  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = mode_ordering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse:
        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
        break;
    }
  }
  return t;
}
void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free(t->indices[i]);
  }
  free(t->indices);
  free(t->dimensions);
  free(t->mode_ordering);
  free(t->mode_types);
  free(t);
}
#endif

int compute(taco_tensor_t *t, taco_tensor_t *f, taco_tensor_t *s) {
  int32_t* restrict t1_pos = (int32_t*)(t->indices[0][0]);
  uint16_t* restrict t1_rle = (uint16_t*)(t->indices[0][1]);
  double* restrict t_vals = (double*)(t->vals);
  int32_t* restrict f1_pos = (int32_t*)(f->indices[0][0]);
  uint16_t* restrict f1_rle = (uint16_t*)(f->indices[0][1]);
  double* restrict f_vals = (double*)(f->vals);
  int32_t* restrict s1_pos = (int32_t*)(s->indices[0][0]);
  uint16_t* restrict s1_rle = (uint16_t*)(s->indices[0][1]);
  double* restrict s_vals = (double*)(s->vals);

  t1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
  t1_pos[0] = 0;
  int32_t t1_rle_size = 33554432;
  t1_rle = (uint16_t*)malloc(sizeof(uint16_t) * t1_rle_size);
  int32_t itpos = 0;
  int32_t t_capacity = 33554432;
  t_vals = (double*)malloc(sizeof(double) * t_capacity);


  /* mergers */
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter */
  int32_t ifpos = f1_pos[0];
  int32_t pf1_end = f1_pos[1];
  int32_t ifcoord = 0;
  int32_t f1_remaining_count = f1_rle[ifpos];
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter end */
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter */
  int32_t ispos = s1_pos[0];
  int32_t ps1_end = s1_pos[1];
  int32_t iscoord = 0;
  int32_t s1_remaining_count = s1_rle[ispos];
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter end */
  /* rangers */

  while (ifpos < pf1_end && ispos < ps1_end) {
    int32_t locDist = 1;
    int32_t locCount = TACO_MIN(f1_remaining_count,s1_remaining_count);
    if (t_capacity <= itpos) {
      t_vals = (double*)realloc(t_vals, sizeof(double) * (t_capacity * 2));
      t_capacity *= 2;
    }
    t_vals[itpos] = f_vals[ifpos] + s_vals[ispos];
    t1_rle[itpos] = 1;
    itpos = itpos;
    itpos++;
    ifcoord++;
    f1_remaining_count = f1_remaining_count - 1;
    if (f1_remaining_count == 0) {
      ifpos++;
      f1_remaining_count = f1_rle[ifpos];
    }
    iscoord++;
    s1_remaining_count = s1_remaining_count - 1;
    if (s1_remaining_count == 0) {
      ispos++;
      s1_remaining_count = s1_rle[ispos];
    }
    if (locCount > locDist) {
      locCount = locCount - locDist;
      int32_t t1_rle_repeat = locCount + t1_rle[(itpos + -1)];
      int32_t t1_rle_store_bound = t1_rle_repeat / 65535;
      int32_t t1_rle_left_over = t1_rle_repeat % 65535;
      int32_t t1_rle_loop_start = 0;
      if (t1_rle_size <= itpos + t1_rle_store_bound) {
        t1_rle = (uint16_t*)realloc(t1_rle, sizeof(uint16_t) * (t1_rle_size * 2));
        t1_rle_size *= 2;
      }
      if (t_capacity <= itpos + t1_rle_store_bound) {
        t_vals = (double*)realloc(t_vals, sizeof(double) * (t_capacity * 2));
        t_capacity *= 2;
      }
      if (t1_rle_left_over == 0) {
        t1_rle_loop_start = 1;
        t1_rle[itpos + -1] = 65535;
      }
      for (int32_t t1_rle_store = t1_rle_loop_start; t1_rle_store < t1_rle_store_bound; t1_rle_store++) {
        t1_rle[itpos + (-1 + (t1_rle_store + 1))] = 65535;
        t_vals[itpos + ((-1 + t1_rle_store) + 1)] = t_vals[(itpos + -1)];
      }
      if (t1_rle_repeat % 65535 > 0)
        t1_rle[itpos + -1] = t1_rle_repeat % 65535;

      itpos += t1_rle_store_bound;
      f1_remaining_count = f1_remaining_count - locCount;
      ifcoord += locCount;
      if (f1_remaining_count == 0) {
        f1_remaining_count = f1_rle[(ifpos + 1)];
        ifpos++;
      }
      s1_remaining_count = s1_remaining_count - locCount;
      iscoord += locCount;
      if (s1_remaining_count == 0) {
        s1_remaining_count = s1_rle[(ispos + 1)];
        ispos++;
      }
    }
  }

  t1_pos[1] = itpos;

  t->indices[0][0] = (uint8_t*)(t1_pos);
  t->indices[0][1] = (uint8_t*)(t1_rle);
  t->vals = (uint8_t*)t_vals;
  return 0;
}

int assemble(taco_tensor_t *t, taco_tensor_t *f, taco_tensor_t *s) {
  int32_t* restrict t1_pos = (int32_t*)(t->indices[0][0]);
  uint16_t* restrict t1_rle = (uint16_t*)(t->indices[0][1]);
  double* restrict t_vals = (double*)(t->vals);
  int32_t* restrict f1_pos = (int32_t*)(f->indices[0][0]);
  uint16_t* restrict f1_rle = (uint16_t*)(f->indices[0][1]);
  int32_t* restrict s1_pos = (int32_t*)(s->indices[0][0]);
  uint16_t* restrict s1_rle = (uint16_t*)(s->indices[0][1]);

  t1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
  t1_pos[0] = 0;
  int32_t t1_rle_size = 33554432;
  t1_rle = (uint16_t*)malloc(sizeof(uint16_t) * t1_rle_size);
  int32_t itpos = 0;


  /* mergers */
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter */
  int32_t ifpos = f1_pos[0];
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter end */
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter */
  int32_t ispos = s1_pos[0];
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter end */
  /* rangers */

  t1_pos[1] = itpos;

  t_vals = (double*)malloc(sizeof(double) * itpos);

  t->indices[0][0] = (uint8_t*)(t1_pos);
  t->indices[0][1] = (uint8_t*)(t1_rle);
  t->vals = (uint8_t*)t_vals;
  return 0;
}

int evaluate(taco_tensor_t *t, taco_tensor_t *f, taco_tensor_t *s) {
  int32_t* restrict t1_pos = (int32_t*)(t->indices[0][0]);
  uint16_t* restrict t1_rle = (uint16_t*)(t->indices[0][1]);
  double* restrict t_vals = (double*)(t->vals);
  int32_t* restrict f1_pos = (int32_t*)(f->indices[0][0]);
  uint16_t* restrict f1_rle = (uint16_t*)(f->indices[0][1]);
  double* restrict f_vals = (double*)(f->vals);
  int32_t* restrict s1_pos = (int32_t*)(s->indices[0][0]);
  uint16_t* restrict s1_rle = (uint16_t*)(s->indices[0][1]);
  double* restrict s_vals = (double*)(s->vals);

  t1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
  t1_pos[0] = 0;
  int32_t t1_rle_size = 33554432;
  t1_rle = (uint16_t*)malloc(sizeof(uint16_t) * t1_rle_size);
  int32_t itpos = 0;
  int32_t t_capacity = 33554432;
  t_vals = (double*)malloc(sizeof(double) * t_capacity);


  /* mergers */
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter */
  int32_t ifpos = f1_pos[0];
  int32_t pf1_end = f1_pos[1];
  int32_t ifcoord = 0;
  int32_t f1_remaining_count = f1_rle[ifpos];
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter end */
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter */
  int32_t ispos = s1_pos[0];
  int32_t ps1_end = s1_pos[1];
  int32_t iscoord = 0;
  int32_t s1_remaining_count = s1_rle[ispos];
  /* [LowererImpl::codeToInitializeIteratorVar] repeatIter end */
  /* rangers */

  while (ifpos < pf1_end && ispos < ps1_end) {
    int32_t locDist = 1;
    int32_t locCount = TACO_MIN(f1_remaining_count,s1_remaining_count);
    if (t_capacity <= itpos) {
      t_vals = (double*)realloc(t_vals, sizeof(double) * (t_capacity * 2));
      t_capacity *= 2;
    }
    t_vals[itpos] = f_vals[ifpos] + s_vals[ispos];
    t1_rle[itpos] = 1;
    itpos = itpos;
    itpos++;
    ifcoord++;
    f1_remaining_count = f1_remaining_count - 1;
    if (f1_remaining_count == 0) {
      ifpos++;
      f1_remaining_count = f1_rle[ifpos];
    }
    iscoord++;
    s1_remaining_count = s1_remaining_count - 1;
    if (s1_remaining_count == 0) {
      ispos++;
      s1_remaining_count = s1_rle[ispos];
    }
    if (locCount > locDist) {
      locCount = locCount - locDist;
      int32_t t1_rle_repeat = locCount + t1_rle[(itpos + -1)];
      int32_t t1_rle_store_bound = t1_rle_repeat / 65535;
      int32_t t1_rle_left_over = t1_rle_repeat % 65535;
      int32_t t1_rle_loop_start = 0;
      if (t1_rle_size <= itpos + t1_rle_store_bound) {
        t1_rle = (uint16_t*)realloc(t1_rle, sizeof(uint16_t) * (t1_rle_size * 2));
        t1_rle_size *= 2;
      }
      if (t_capacity <= itpos + t1_rle_store_bound) {
        t_vals = (double*)realloc(t_vals, sizeof(double) * (t_capacity * 2));
        t_capacity *= 2;
      }
      if (t1_rle_left_over == 0) {
        t1_rle_loop_start = 1;
        t1_rle[itpos + -1] = 65535;
      }
      for (int32_t t1_rle_store = t1_rle_loop_start; t1_rle_store < t1_rle_store_bound; t1_rle_store++) {
        t1_rle[itpos + (-1 + (t1_rle_store + 1))] = 65535;
        t_vals[itpos + ((-1 + t1_rle_store) + 1)] = t_vals[(itpos + -1)];
      }
      if (t1_rle_repeat % 65535 > 0)
        t1_rle[itpos + -1] = t1_rle_repeat % 65535;

      itpos += t1_rle_store_bound;
      f1_remaining_count = f1_remaining_count - locCount;
      ifcoord += locCount;
      if (f1_remaining_count == 0) {
        f1_remaining_count = f1_rle[(ifpos + 1)];
        ifpos++;
      }
      s1_remaining_count = s1_remaining_count - locCount;
      iscoord += locCount;
      if (s1_remaining_count == 0) {
        s1_remaining_count = s1_rle[(ispos + 1)];
        ispos++;
      }
    }
  }

  t1_pos[1] = itpos;

  t->indices[0][0] = (uint8_t*)(t1_pos);
  t->indices[0][1] = (uint8_t*)(t1_rle);
  t->vals = (uint8_t*)t_vals;
  return 0;
}

/*
 * The `pack` functions convert coordinate and value arrays in COO format,
 * with nonzeros sorted lexicographically by their coordinates, to the
 * specified input format.
 *
 * The `unpack` function converts the specified output format to coordinate
 * and value arrays in COO format.
 *
 * For both, the `_COO_pos` arrays contain two elements, where the first is 0
 * and the second is the number of nonzeros in the tensor.
 */

int pack_f(taco_tensor_t *f, int* f_COO1_pos, int* f_COO1_crd, double* f_COO_vals) {
  int32_t* restrict f1_pos = (int32_t*)(f->indices[0][0]);
  uint16_t* restrict f1_rle = (uint16_t*)(f->indices[0][1]);
  double* restrict f_vals = (double*)(f->vals);

  f1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
  f1_pos[0] = 0;
  int32_t f1_rle_size = 33554432;
  f1_rle = (uint16_t*)malloc(sizeof(uint16_t) * f1_rle_size);
  int32_t ifpos = 0;
  int32_t f_capacity = 33554432;
  f_vals = (double*)malloc(sizeof(double) * f_capacity);


  /* mergers */
  int32_t if_COOpos = f_COO1_pos[0];
  int32_t pf_COO1_end = f_COO1_pos[1];
  /* rangers */

  while (if_COOpos < pf_COO1_end) {
    /* [lowerMergePoint] loadPosIterCoordinates */
    /* [lowerMergePoint] resolvedCoordinate */
    int32_t i = f_COO1_crd[if_COOpos];
    /* [lowerMergePoint] loadLocatorPosVars */
    /* [lowerMergePoint] deduplicationLoops */
    double f_COO_val = f_COO_vals[if_COOpos];
    if_COOpos++;
    while (if_COOpos < pf_COO1_end && f_COO1_crd[if_COOpos] == i) {
      f_COO_val += f_COO_vals[if_COOpos];
      if_COOpos++;
    }
    /* [lowerMergePoint] caseStmts */
    if (f_capacity <= ifpos) {
      f_vals = (double*)realloc(f_vals, sizeof(double) * (f_capacity * 2));
      f_capacity *= 2;
    }
    f_vals[ifpos] = f_COO_val;
    f1_rle[ifpos] = 1;
    ifpos = ifpos;
    ifpos++;
    /* [lowerMergePoint] incIteratorVarStmts */
  }

  f1_pos[1] = ifpos;

  f->indices[0][0] = (uint8_t*)(f1_pos);
  f->indices[0][1] = (uint8_t*)(f1_rle);
  f->vals = (uint8_t*)f_vals;
  return 0;
}

int pack_s(taco_tensor_t *s, int* s_COO1_pos, int* s_COO1_crd, double* s_COO_vals) {
  int32_t* restrict s1_pos = (int32_t*)(s->indices[0][0]);
  uint16_t* restrict s1_rle = (uint16_t*)(s->indices[0][1]);
  double* restrict s_vals = (double*)(s->vals);

  s1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
  s1_pos[0] = 0;
  int32_t s1_rle_size = 33554432;
  s1_rle = (uint16_t*)malloc(sizeof(uint16_t) * s1_rle_size);
  int32_t ispos = 0;
  int32_t s_capacity = 33554432;
  s_vals = (double*)malloc(sizeof(double) * s_capacity);


  /* mergers */
  int32_t is_COOpos = s_COO1_pos[0];
  int32_t ps_COO1_end = s_COO1_pos[1];
  /* rangers */

  while (is_COOpos < ps_COO1_end) {
    /* [lowerMergePoint] loadPosIterCoordinates */
    /* [lowerMergePoint] resolvedCoordinate */
    int32_t i = s_COO1_crd[is_COOpos];
    /* [lowerMergePoint] loadLocatorPosVars */
    /* [lowerMergePoint] deduplicationLoops */
    double s_COO_val = s_COO_vals[is_COOpos];
    is_COOpos++;
    while (is_COOpos < ps_COO1_end && s_COO1_crd[is_COOpos] == i) {
      s_COO_val += s_COO_vals[is_COOpos];
      is_COOpos++;
    }
    /* [lowerMergePoint] caseStmts */
    if (s_capacity <= ispos) {
      s_vals = (double*)realloc(s_vals, sizeof(double) * (s_capacity * 2));
      s_capacity *= 2;
    }
    s_vals[ispos] = s_COO_val;
    s1_rle[ispos] = 1;
    ispos = ispos;
    ispos++;
    /* [lowerMergePoint] incIteratorVarStmts */
  }

  s1_pos[1] = ispos;

  s->indices[0][0] = (uint8_t*)(s1_pos);
  s->indices[0][1] = (uint8_t*)(s1_rle);
  s->vals = (uint8_t*)s_vals;
  return 0;
}

int unpack(int** t_COO1_pos_ptr, int** t_COO1_crd_ptr, double** t_COO_vals_ptr, taco_tensor_t *t) {
  int* t_COO1_pos;
  int* t_COO1_crd;
  double* t_COO_vals;
  int32_t* restrict t1_pos = (int32_t*)(t->indices[0][0]);
  uint16_t* restrict t1_rle = (uint16_t*)(t->indices[0][1]);
  double* restrict t_vals = (double*)(t->vals);

  t_COO1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
  t_COO1_pos[0] = 0;
  int32_t t_COO1_crd_size = 33554432;
  t_COO1_crd = (int32_t*)malloc(sizeof(int32_t) * t_COO1_crd_size);
  int32_t it_COOpos = 0;
  int32_t t_COO_capacity = 33554432;
  t_COO_vals = (double*)malloc(sizeof(double) * t_COO_capacity);


  int32_t i = 0;

  for (int32_t itpos = t1_pos[0]; itpos < t1_pos[1]; itpos++) {
    int32_t t1_pos_off = 0;
    int32_t t1_pos_save = itpos;
    for (int32_t t1_rep_iter = 0; t1_rep_iter < t1_rle[itpos]; t1_rep_iter++) {
      t1_pos_off = 0;
      itpos += t1_pos_off;
      if (t_COO_capacity <= it_COOpos) {
        t_COO_vals = (double*)realloc(t_COO_vals, sizeof(double) * (t_COO_capacity * 2));
        t_COO_capacity *= 2;
      }
      t_COO_vals[it_COOpos] = t_vals[itpos];
      if (t_COO1_crd_size <= it_COOpos) {
        t_COO1_crd = (int32_t*)realloc(t_COO1_crd, sizeof(int32_t) * (t_COO1_crd_size * 2));
        t_COO1_crd_size *= 2;
      }
      t_COO1_crd[it_COOpos] = i;
      it_COOpos++;
      t_COO1_pos[1] = it_COOpos;
      i++;
    }
    itpos = t1_pos_save;
  }

  *t_COO1_pos_ptr = t_COO1_pos;
  *t_COO1_crd_ptr = t_COO1_crd;
  *t_COO_vals_ptr = t_COO_vals;
  return 0;
}
