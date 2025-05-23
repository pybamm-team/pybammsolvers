#ifndef PYBAMM_SUNDIALS_FUNCTIONS_HPP
#define PYBAMM_SUNDIALS_FUNCTIONS_HPP

#include "common.hpp"

template<typename T>
void axpy(int const n, T alpha, const T* x, T* y) {
  if (!x || !y) {
      return;
  }
  for (int i=0; i<n; ++i) {
      *y++ += alpha**x++;
  }
}

int residual_eval(sunrealtype tres, N_Vector yy, N_Vector yp, N_Vector rr,
                    void *user_data);

int jtimes_eval(sunrealtype tt, N_Vector yy, N_Vector yp, N_Vector rr,
                  N_Vector v, N_Vector Jv, sunrealtype cj, void *user_data,
                  N_Vector tmp1, N_Vector tmp2);

int events_eval(sunrealtype t, N_Vector yy, N_Vector yp, sunrealtype *events_ptr,
                  void *user_data);

int sensitivities_eval(int Ns, sunrealtype t, N_Vector yy, N_Vector yp,
                         N_Vector resval, N_Vector *yS, N_Vector *ypS,
                         N_Vector *resvalS, void *user_data, N_Vector tmp1,
                         N_Vector tmp2, N_Vector tmp3);

int jacobian_eval(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp,
                    N_Vector resvec, SUNMatrix JJ, void *user_data,
                    N_Vector tempv1, N_Vector tempv2, N_Vector tempv3);

int residual_eval_approx(sunindextype Nlocal, sunrealtype tt, N_Vector yy,
                           N_Vector yp, N_Vector gval, void *user_data);

#include "sundials_functions.inl"

#endif // PYBAMM_SUNDIALS_FUNCTIONS_HPP
