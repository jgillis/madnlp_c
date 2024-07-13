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
  size_t nw;
  /// @brief number of equality constraints
  size_t nc;

  size_t* nzj_i; // 1-based
  size_t* nzj_j;
  size_t* nzh_i;
  size_t* nzh_j;

  size_t nnzj;
  size_t nnzh;
  size_t nnzo;

  void* user_data;
};

struct MadnlpCNumericIn {
  const double* x0;
  const double* l0;
  const double* ubx;
  const double* lbx;
  const double* ubg;
  const double* lbg;
};

struct MadnlpCNumericOut {
  double* sol;
  double* con;
  double* obj;
  double* mul;
  double* mul_L;
  double* mul_U;

};

MADNLP_SYMBOL_EXPORT struct MadnlpCSolver* madnlp_c_create(struct MadnlpCInterface* nlp_interface);
MADNLP_SYMBOL_EXPORT madnlp_int madnlp_c_solve(struct MadnlpCSolver*, struct MadnlpCNumericIn* in, struct MadnlpCNumericOut* out);

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
