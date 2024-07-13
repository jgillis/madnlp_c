#ifndef _MADNLP_C_H
#define _MADNLP_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Symbol visibility in DLLs */
#ifndef MADNLP_SYMBOL_EXPORT
#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#if defined(STATIC_LINKED)
#define MADNLP_SYMBOL_EXPORT
#else
#define MADNLP_SYMBOL_EXPORT __declspec(dllexport)
#endif
#elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#define MADNLP_SYMBOL_EXPORT __attribute__((visibility("default")))
#else
#define MADNLP_SYMBOL_EXPORT
#endif
#endif


#define madnlp_int int
#define madnlp_real double

// structs
struct MadnlpCStats;
struct MadnlpCDims;
struct MadnlpCInterface;

// Opaque types
typedef struct MadnlpCSolver MadnlpCSolver;

// function pointer types
typedef madnlp_int (*MadnlpCEvalObj)(const double*, double *, void*);
typedef madnlp_int (*MadnlpCEvalConstr)(const double*, double *, void*);
typedef madnlp_int (*MadnlpCEvalObjGrad)(const double*, double*, void*);
typedef madnlp_int (*MadnlpCEvalConstrJac)(const double*, double*, void*);
typedef madnlp_int (*MadnlpCEvalLagHess)(double, const double*, const double*, double*, void*);

struct MadnlpCStats {
  madnlp_int iter;
};

struct MadnlpCInterface {
  MadnlpCEvalObj eval_obj;
  MadnlpCEvalConstr eval_constr;
  MadnlpCEvalObjGrad eval_obj_grad;
  MadnlpCEvalConstrJac eval_constr_jac;
  MadnlpCEvalLagHess eval_lag_hess;

  /// @brief number of variables
  madnlp_int nw;
  /// @brief number of equality constraints
  madnlp_int nc;

  madnlp_int* nzj_i; // 1-based
  madnlp_int* nzj_j;
  madnlp_int* nzh_i;
  madnlp_int* nzh_j;

  madnlp_int nnzj;
  madnlp_int nnzh;
  madnlp_int nnzo;

  void* user_data;
};

struct MadnlpCNumericIn {
  double* x0;
  double* l0;
  double* lbx;
  double* ubx;
  double* lbg;
  double* ubg;
};

struct MadnlpCNumericOut {
  const double* sol;
  const double* con;
  const double* obj;
  const double* mul;
  const double* mul_L;
  const double* mul_U;
};

MADNLP_SYMBOL_EXPORT struct MadnlpCSolver* madnlp_c_create(struct MadnlpCInterface* nlp_interface);
MADNLP_SYMBOL_EXPORT const struct MadnlpCNumericIn* madnlp_c_input(struct MadnlpCSolver*);
MADNLP_SYMBOL_EXPORT const struct MadnlpCNumericOut* madnlp_c_output(struct MadnlpCSolver*);
MADNLP_SYMBOL_EXPORT madnlp_int madnlp_c_solve(struct MadnlpCSolver*);

/* -1 for not found, 0 for double, 1 for int, 2 for bool, 3 for string */
MADNLP_SYMBOL_EXPORT int madnlp_c_option_type(const char* name);
MADNLP_SYMBOL_EXPORT int madnlp_c_set_option_double(struct MadnlpCSolver* s, const char* name, double val);
MADNLP_SYMBOL_EXPORT int madnlp_c_set_option_bool(struct MadnlpCSolver* s, const char* name, int val);
MADNLP_SYMBOL_EXPORT int madnlp_c_set_option_int(struct MadnlpCSolver* s, const char* name, int val);
MADNLP_SYMBOL_EXPORT int madnlp_c_set_option_string(struct MadnlpCSolver* s, const char* name, const char* val);

MADNLP_SYMBOL_EXPORT const struct MadnlpCStats* madnlp_c_get_stats(struct MadnlpCSolver* s);
MADNLP_SYMBOL_EXPORT void madnlp_c_destroy(struct MadnlpCSolver*);

#ifdef __cplusplus
}
#endif

#endif // _MADNLP_C_H
