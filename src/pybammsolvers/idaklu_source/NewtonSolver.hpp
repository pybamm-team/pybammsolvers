#ifndef PYBAMM_NEWTON_SOLVER_HPP
#define PYBAMM_NEWTON_SOLVER_HPP

#include "common.hpp"
#include "Options.hpp"
#include "SolverLog.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <stdexcept>

enum class NewtonSolveType {
  DECOUPLED_FULL,       // zero diff residuals, reuse IDA's LS
  DECOUPLED_SUBBLOCK,   // n_alg x n_alg sub-block with own LS
  COUPLED_FULL          // full IDACalcIC-style with id-vector step split
};

enum class NewtonResult {
  CONVERGED_WRMS_AND_STEPTOL,      // WRMS(delta,ewt) <= epsNewt AND step_tol satisfied
  CONVERGED_WRMS_STEP_DIVERGED,    // WRMS converged but step increased residual (reverted)
  CONVERGED_WRMS_AT_MAX_ITER,      // WRMS converged during loop but hit max_iter before steptol
  LSETUP_FAIL,                     // SUNLinSolSetup failed
  LSOLVE_FAIL,                     // SUNLinSolSolve failed
  MAX_ITER_NO_CONVERGE,            // max_iter reached, WRMS never satisfied
};

inline bool newton_success(NewtonResult r) {
  return r == NewtonResult::CONVERGED_WRMS_AND_STEPTOL ||
         r == NewtonResult::CONVERGED_WRMS_STEP_DIVERGED ||
         r == NewtonResult::CONVERGED_WRMS_AT_MAX_ITER;
}

inline const char* newton_result_reason(NewtonResult r) {
  switch (r) {
    case NewtonResult::CONVERGED_WRMS_AND_STEPTOL:    return "wrms+steptol";
    case NewtonResult::CONVERGED_WRMS_STEP_DIVERGED:  return "wrms (step diverged, reverted)";
    case NewtonResult::CONVERGED_WRMS_AT_MAX_ITER:    return "wrms (max_iter)";
    case NewtonResult::LSETUP_FAIL:                   return "lsetup fail";
    case NewtonResult::LSOLVE_FAIL:                   return "lsolve fail";
    case NewtonResult::MAX_ITER_NO_CONVERGE:          return "max_iter, no convergence";
  }
  return "unknown";
}

/**
 * @brief Newton solver for consistent initial conditions of DAE systems.
 *
 * Supports three solve types determined at construction time based on mass
 * matrix structure and user-selected mode:
 *
 *  - DECOUPLED_FULL: Solves the full n_states system with differential
 *    residuals zeroed and differential Jacobian rows/cols set to identity,
 *    reusing IDA's linear solver (supports preconditioners).
 *
 *  - DECOUPLED_SUBBLOCK: Solves only the n_alg x n_alg algebraic sub-block
 *    with a separate direct linear solver using dedicated algebraic residual
 *    and Jacobian functions.
 *
 *  - COUPLED_FULL: Solves the full coupled system for y_alg and ydot_diff
 *    simultaneously (mirroring IDACalcIC IDA_YA_YDP_INIT). Used when mass
 *    matrix has nonzeros in algebraic rows.
 */
template <class ExprSet>
class NewtonSolver {
public:
  NewtonSolver(
    ExprSet* functions,
    int len_rhs,
    int len_alg,
    const sunrealtype* id_val,
    const SetupOptions& setup_opts,
    const SolverOptions& solver_opts,
    SUNContext sunctx,
    SUNLinearSolver LS_ida,
    SUNMatrix J_ida,
    N_Vector yy,
    N_Vector yyp,
    N_Vector id,
    const std::vector<int64_t>& jac_colptrs,
    const std::vector<int64_t>& jac_rowvals,
    sunrealtype rtol,
    N_Vector avtol,
    void* ida_mem
  );

  ~NewtonSolver();

  NewtonResult solve(sunrealtype t, sunrealtype t_next, SolverLog& log);

  bool is_decoupled() const {
    return solve_type_ != NewtonSolveType::COUPLED_FULL;
  }

  NewtonSolveType solve_type() const { return solve_type_; }

private:
  NewtonResult solve_decoupled_full(sunrealtype t, sunrealtype t_next, SolverLog& log);
  NewtonResult solve_decoupled_subblock(sunrealtype t, SolverLog& log);
  NewtonResult solve_coupled_full(sunrealtype t, sunrealtype t_next, SolverLog& log);

  void EvalRhsAlg(sunrealtype t, sunrealtype* res_out);
  void EvalAlgRes(sunrealtype t, sunrealtype* res_out);
  void EvalAlgJac(sunrealtype t, sunrealtype* jac_out);
  void EvalFullResidual(sunrealtype t);
  void EvalFullJacobian(sunrealtype t, sunrealtype cj);
  void ZeroDiffRowsColsJacobian();
  void CopySparsityToJIda();
  void CopySparsityToJAlg();

  // WRMS norm helpers (ewt_ must be populated before calling)
  sunrealtype WrmsNormAlg(const sunrealtype* vals) const;
  sunrealtype WrmsNormAlgCompact(const sunrealtype* vals, int n) const;

  // ATimes callbacks for iterative solvers (matrix-free Newton)
  static int newton_atimes_decoupled(void* data, N_Vector v, N_Vector z);
  static int newton_atimes_full(void* data, N_Vector v, N_Vector z);
  int ComputeJv(N_Vector v, N_Vector Jv);
  void SetupATimes();
  void RestoreATimes();

  bool CheckMassMatrixAlignment(const sunrealtype* id_val);
  void PrecomputeSubBlockSparsity(
    const std::vector<int64_t>& jac_colptrs,
    const std::vector<int64_t>& jac_rowvals
  );

  ExprSet* functions_;
  const SetupOptions& setup_opts_;
  const SolverOptions& solver_opts_;
  SUNContext sunctx_;
  NewtonSolveType solve_type_;

  sunrealtype rtol_;
  N_Vector avtol_;  // borrowed, not owned

  int len_rhs_;
  int len_alg_;
  int n_states_;

  // Full-system mode resources (borrowed from IDA, NOT owned)
  SUNLinearSolver LS_ida_;
  SUNMatrix J_ida_;
  N_Vector yy_;
  N_Vector yyp_;
  N_Vector id_;

  // Iterative solver support (matrix-free ATimes)
  void* ida_mem_;
  sunrealtype newton_cj_;
  std::vector<sunrealtype> atimes_tmp_;       // scratch for mass_action result in ComputeJv
  std::vector<sunrealtype> atimes_v_save_;    // scratch for saving diff components of v

  // Sub-block mode resources (owned)
  SUNMatrix J_alg_;
  SUNLinearSolver LS_alg_;
  N_Vector res_alg_vec_;
  N_Vector delta_alg_vec_;

  // Precomputed sub-block sparsity mapping
  int alg_nnz_;
  std::vector<int64_t> alg_colptrs_;
  std::vector<int64_t> alg_rowvals_;
  std::vector<int> alg_data_indices_;

  // Full-system temporary buffers
  N_Vector res_full_vec_;
  N_Vector delta_full_vec_;
  std::vector<sunrealtype> res_full_buf_;

  // Pre-allocated buffers for Newton iteration
  std::vector<sunrealtype> y_iter_save_;     // save y before each step (for revert)
  std::vector<sunrealtype> yp_iter_save_;    // save yp before each step (coupled only)
  std::vector<sunrealtype> y0_save_;         // save initial state for hic retries
  std::vector<sunrealtype> yp0_save_;        // save initial state for hic retries
  std::vector<sunrealtype> full_jac_buf_;    // temp buffer for jac_alg output
  std::vector<sunrealtype> ewt_;             // error weight vector for WRMS norm
};

#include "NewtonSolver.inl"

#endif // PYBAMM_NEWTON_SOLVER_HPP
