#include "madnlp_c.h"
#include <stdio.h>
#include <algorithm>
#include <iostream>

extern "C" {
  int init_julia(int, char**);
  void shutdown_julia(int);
}

double a = 1.0;
double b = 100.0;

int eval_f(const double* w,double* f, void* user_data) {
    f[0] = (a-w[0])*(a-w[0]) + b*(w[1]-w[0]*w[0])*(w[1]-w[0]*w[0]);
    return 0;
}
int eval_g(const double* w,double* c, void* user_data) {
    c[0] = w[0]*w[0]+w[1]*w[1]-1;
    return 0;
}
int eval_grad_f(const double* w,double* g, void* user_data) {
    g[0] = -4*b*w[0]*(w[1]-w[0]*w[0])-2*(a-w[0]);
    g[1] = b*2*(w[1]-w[0]*w[0]);
    return 0;
}
int eval_jac_g(const double *w,double *j, void* user_data) {
    j[0] = 2*w[0];
    j[1] = 2*w[1];
    return 0;
}
int eval_h(double obj_scale, const double* w, const double* l, double* h, void* user_data) {
    // dw0dw0
    h[0] = +2 -4*b*w[1] +12*b*w[0]*w[0];
    // dw0dw1|dw1dw0
    h[1] = -4*b*w[0];
    // dw1dw1
    h[2] = 2*b;

    // constraints 
    h[0] += l[0]*2;
    h[2] += l[0]*2;

    return 0;
}

int main(int argc, char** argv) {
  init_julia(argc, argv);

  madnlp_int nw = 2;
  madnlp_int nc = 1;

  MadnlpCInterface interf;
  interf.eval_obj = eval_f;
  interf.eval_constr = eval_g;
  interf.eval_constr_jac = eval_jac_g;
  interf.eval_obj_grad = eval_grad_f;
  interf.eval_lag_hess = eval_h;

  interf.nw = nw;
  interf.nc = nc;
  madnlp_int nzj_i[2] = {1,1};
  madnlp_int nzj_j[2] = {1,2};
  madnlp_int nzh_i[3] = {1,1,2};
  madnlp_int nzh_j[3] = {1,2,2};
  interf.nzj_i = nzj_i;
  interf.nzj_j = nzj_j;
  interf.nzh_i = nzh_i;
  interf.nzh_j = nzh_j;

  interf.nnzj = 2;
  interf.nnzh = 3;

  double x0[2] = {1,1};
  double l0[1] = {1};
  double lbx[2] = {-100,-100};
  double ubx[2] = {100,100};
  double lbg[1] = {0};
  double ubg[1] = {0};

  printf("lbx %p, ubx %p\n", lbx, ubx);

  printf("interf %p\n", &interf);

  struct MadnlpCSolver* solver = madnlp_c_create(&interf);

  madnlp_c_set_option_int(solver, "max_iter", 5);
  madnlp_c_set_option_int(solver, "print_level", 2);
  madnlp_c_set_option_int(solver, "lin_solver_id", 1);

  const MadnlpCNumericIn* in = madnlp_c_input(solver);
  std::copy(x0,x0+2,in->x0);
  std::copy(l0,l0+1,in->l0);
  std::copy(lbx,lbx+2,in->lbx);
  std::copy(ubx,ubx+2,in->ubx);
  std::copy(lbg,lbg+1,in->lbg);
  std::copy(ubg,ubg+1,in->ubg);

  madnlp_c_solve(solver);
  const MadnlpCNumericOut* out = madnlp_c_output(solver);

  double sol[2];
  double cons[1];
  double obj[1];
  std::copy(out->sol,out->sol+2,sol);
  std::copy(out->con,out->con+1,cons);
  std::copy(out->obj,out->obj+1,obj);

  std::cout << "sol: " << sol[0] << ", " << sol[1] << std::endl;
  std::cout << "obj: " << obj[1] << std::endl;
  std::cout << "con: " << con[1] << std::endl;

  shutdown_julia(0);

  return 0;
}
