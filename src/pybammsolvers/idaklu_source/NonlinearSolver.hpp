#ifndef PYBAMM_NONLINEAR_SOLVER_HPP
#define PYBAMM_NONLINEAR_SOLVER_HPP

#include "common.hpp"
#include "LinearSolver.hpp"
#include "Expressions/Casadi/CasadiFunctions.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <functional>
#include <memory>

enum class NonlinearResult {
  CONVERGED_WRMS_AND_STEPTOL,
  CONVERGED_WRMS_STEP_DIVERGED,
  CONVERGED_WRMS_AT_MAX_ITER,
  LSETUP_FAIL,
  LSOLVE_FAIL,
  MAX_ITER_NO_CONVERGE,
};

inline bool nonlinear_success(NonlinearResult r) {
  return r == NonlinearResult::CONVERGED_WRMS_AND_STEPTOL ||
         r == NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED ||
         r == NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER;
}

inline const char* nonlinear_result_reason(NonlinearResult r) {
  switch (r) {
    case NonlinearResult::CONVERGED_WRMS_AND_STEPTOL:    return "wrms+steptol";
    case NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED:  return "wrms (step diverged, reverted)";
    case NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER:    return "wrms (max_iter)";
    case NonlinearResult::LSETUP_FAIL:                   return "lsetup fail";
    case NonlinearResult::LSOLVE_FAIL:                   return "lsolve fail";
    case NonlinearResult::MAX_ITER_NO_CONVERGE:          return "max_iter, no convergence";
  }
  return "unknown";
}

/**
 * @brief Unified nonlinear solver: find x such that F(t, x, p) = 0.
 *
 * Single Newton iteration loop used by both:
 *  - Standalone mode (Python-facing): owns a LinearSolver + CasADi functions.
 *  - Borrowed mode (IDA integration): uses borrowed LinearSolver + callbacks.
 *
 * Zero allocations in the hotpath.
 */
class NonlinearSolver {
public:
  using ResidualFn = std::function<void(sunrealtype t, const sunrealtype* x, sunrealtype* res)>;
  using JacobianFn = std::function<void(sunrealtype t, const sunrealtype* x)>;

  /**
   * @brief Standalone constructor (owns LinearSolver + CasADi functions).
   */
  NonlinearSolver(
    const casadi::Function& residual_fn,
    const casadi::Function& jacobian_fn,
    int n_vars,
    np_array atol_np,
    double rtol,
    double step_tol,
    int inputs_length,
    py::dict options
  );

  /**
   * @brief Borrowed constructor (IDA integration mode).
   * Does not own the LinearSolver or evaluate CasADi functions.
   * The caller provides callbacks for residual/Jacobian evaluation and
   * manages the LinearSolver lifetime.
   */
  NonlinearSolver(
    LinearSolver* ls,
    ResidualFn eval_residual,
    JacobianFn eval_jacobian,
    int n_vars,
    const sunrealtype* atol_data,
    sunrealtype rtol,
    sunrealtype step_tol,
    int max_iter,
    int max_backtracks,
    sunrealtype epsNewt
  );

  ~NonlinearSolver();

  NonlinearSolver(const NonlinearSolver&) = delete;
  NonlinearSolver& operator=(const NonlinearSolver&) = delete;

  /**
   * @brief Single solve: find x such that F(t, x, p) = 0.
   * x is read as initial guess and overwritten with solution (IN-PLACE).
   * Returns the result status.
   */
  NonlinearResult solve_single(sunrealtype t, sunrealtype* x, const sunrealtype* p);

  /**
   * @brief Batch solve over t_eval (Python-facing, loops in C++).
   * x0 is the initial guess; reused as next guess after each solve.
   * out: optional pre-allocated (n_vars, n_times) array for in-place output.
   */
  py::object solve(
    np_array t_eval_np,
    np_array x0_np,
    np_array inputs_np,
    py::object out = py::none()
  );

  const std::string& last_message() const { return last_message_; }
  int last_num_iterations() const { return last_num_iterations_; }

private:
  // THE single Newton loop -- written ONCE, used by both standalone and IDA
  NonlinearResult RunNewtonLoop(sunrealtype t);

  // Evaluate residual and return infinity norm
  sunrealtype EvalResidualAndNorm(sunrealtype t);

  // Evaluate Jacobian, factorize, and solve J*delta = res.
  // Returns 0 on success, 1 for setup fail, -1 for solve fail.
  int SetupAndSolveLinearSystem(sunrealtype t);

  // WRMS norm: sqrt(mean((vals[i] * ewt_[i])^2))
  sunrealtype WrmsNorm(const sunrealtype* vals) const;

  // Infinity norm: max(|vals[i]|)
  sunrealtype InfNorm(const sunrealtype* vals) const;

  void ComputeEwt();
  void SaveIterate();
  void RevertAndApply(sunrealtype alpha);

  int n_vars_;
  sunrealtype rtol_;
  sunrealtype step_tol_;
  int max_iter_;
  int max_backtracks_;
  sunrealtype epsNewt_;

  // LinearSolver (owned or borrowed)
  LinearSolver* ls_;
  std::unique_ptr<LinearSolver> owned_ls_;

  // Residual/Jacobian callbacks (set by both standalone and borrowed modes)
  ResidualFn eval_residual_;
  JacobianFn eval_jacobian_;

  // CasADi function wrappers (standalone only, null for borrowed mode)
  std::unique_ptr<CasadiFunction> residual_casadi_;
  std::unique_ptr<CasadiFunction> jacobian_casadi_;

  // Pre-allocated working vectors (set once, ZERO allocations in hotpath)
  std::vector<sunrealtype> x_;         // current iterate
  std::vector<sunrealtype> res_;       // residual F(t, x, p)
  std::vector<sunrealtype> delta_;     // Newton step (J\F)
  std::vector<sunrealtype> x_save_;    // saved iterate for line search
  std::vector<sunrealtype> ewt_;       // error weight vector
  std::vector<sunrealtype> atol_;      // per-variable absolute tolerance
  std::vector<sunrealtype> inputs_;    // parameter vector p (standalone only)
  std::vector<sunrealtype> jac_buf_;   // raw Jacobian values from evaluator

  // Jacobian sparsity (standalone sparse mode)
  bool use_sparse_;
  int jac_nnz_;

  // Diagnostics
  std::string last_message_;
  int last_num_iterations_;
};

#include "NonlinearSolver.inl"

#endif // PYBAMM_NONLINEAR_SOLVER_HPP
