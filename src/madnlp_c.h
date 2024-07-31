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


#define madnlp_int long long int
#define madnlp_real double

// structs
struct MadnlpCStats;
struct MadnlpCDims;
struct MadnlpCInterface;

// Opaque types
typedef struct MadnlpCSolver MadnlpCSolver;

// function pointer types
typedef int (*MadnlpCEvalObj)(const double*, double *, void*);
typedef int (*MadnlpCEvalConstr)(const double*, double *, void*);
typedef int (*MadnlpCEvalObjGrad)(const double*, double*, void*);
typedef int (*MadnlpCEvalConstrJac)(const double*, double*, void*);
typedef int (*MadnlpCEvalLagHess)(double, const double*, const double*, double*, void*);

enum MadnlpCStatus {
  MADNLP_SOLVE_SUCCEEDED = 1,
  MADNLP_SOLVED_TO_ACCEPTABLE_LEVEL = 2,
  MADNLP_SEARCH_DIRECTION_BECOMES_TOO_SMALL = 3,
  MADNLP_DIVERGING_ITERATES = 4,
  MADNLP_INFEASIBLE_PROBLEM_DETECTED = 5,
  MADNLP_MAXIMUM_ITERATIONS_EXCEEDED = 6,
  MADNLP_MAXIMUM_WALLTIME_EXCEEDED = 7,
  MADNLP_INITIAL = 11,
  MADNLP_REGULAR = 12,
  MADNLP_RESTORE = 13,
  MADNLP_ROBUST  = 14,
  MADNLP_RESTORATION_FAILED = -1,
  MADNLP_INVALID_NUMBER_DETECTED = -2,
  MADNLP_ERROR_IN_STEP_COMPUTATION = -3,
  MADNLP_NOT_ENOUGH_DEGREES_OF_FREEDOM = -4,
  MADNLP_USER_REQUESTED_STOP = -5,
  MADNLP_INTERNAL_ERROR = -6,
  MADNLP_INVALID_NUMBER_OBJECTIVE = -7,
  MADNLP_INVALID_NUMBER_GRADIENT = -8,
  MADNLP_INVALID_NUMBER_CONSTRAINTS = -9,
  MADNLP_INVALID_NUMBER_JACOBIAN = -10,
  MADNLP_INVALID_NUMBER_HESSIAN_LAGRANGIAN = -11
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

struct MadnlpCStats {
  madnlp_int iter;
  madnlp_int status;
  double dual_feas;
  double primal_feas;
};

MADNLP_SYMBOL_EXPORT struct MadnlpCSolver* madnlp_c_create(struct MadnlpCInterface* nlp_interface);

MADNLP_SYMBOL_EXPORT const struct MadnlpCNumericIn* madnlp_c_input(struct MadnlpCSolver*);

/* -1 for not found, 0 for double, 1 for int, 2 for bool, 3 for string */
MADNLP_SYMBOL_EXPORT int madnlp_c_option_type(const char* name);
MADNLP_SYMBOL_EXPORT int madnlp_c_set_option_double(struct MadnlpCSolver* s, const char* name, double val);
MADNLP_SYMBOL_EXPORT int madnlp_c_set_option_bool(struct MadnlpCSolver* s, const char* name, madnlp_int val);
MADNLP_SYMBOL_EXPORT int madnlp_c_set_option_int(struct MadnlpCSolver* s, const char* name, madnlp_int val);
MADNLP_SYMBOL_EXPORT int madnlp_c_set_option_string(struct MadnlpCSolver* s, const char* name, const char* val);

MADNLP_SYMBOL_EXPORT int madnlp_c_solve(struct MadnlpCSolver*);

MADNLP_SYMBOL_EXPORT const struct MadnlpCNumericOut* madnlp_c_output(struct MadnlpCSolver*);
MADNLP_SYMBOL_EXPORT const struct MadnlpCStats* madnlp_c_get_stats(struct MadnlpCSolver* s);

MADNLP_SYMBOL_EXPORT void madnlp_c_destroy(struct MadnlpCSolver*);

#ifdef __cplusplus
}
#endif

#endif // _MADNLP_C_H