#pragma once

#include "Expressions/Expressions.hpp"
#include "sundials_functions.hpp"

// Constructor

template <class ExprSet>
NewtonSolver<ExprSet>::NewtonSolver(
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
) : functions_(functions),
    setup_opts_(setup_opts),
    solver_opts_(solver_opts),
    sunctx_(sunctx),
    rtol_(rtol),
    avtol_(avtol),
    len_rhs_(len_rhs),
    len_alg_(len_alg),
    n_states_(len_rhs + len_alg),
    LS_ida_(LS_ida),
    J_ida_(J_ida),
    yy_(yy),
    yyp_(yyp),
    id_(id),
    ida_mem_(ida_mem),
    newton_cj_(SUN_RCONST(1.0)),
    J_alg_(nullptr),
    LS_alg_(nullptr),
    res_alg_vec_(nullptr),
    delta_alg_vec_(nullptr),
    alg_nnz_(0),
    res_full_vec_(nullptr),
    delta_full_vec_(nullptr)
{
  DEBUG("NewtonSolver::NewtonSolver (len_rhs=" << len_rhs_ << ", len_alg=" << len_alg_ << ")");

  if (len_alg_ <= 0) {
    solve_type_ = NewtonSolveType::DECOUPLED_FULL;
    return;
  }

  bool mass_aligned = CheckMassMatrixAlignment(id_val);

  if (mass_aligned) {
    if (solver_opts.newton_mode == "algebraic") {
      solve_type_ = NewtonSolveType::DECOUPLED_SUBBLOCK;
    } else {
      solve_type_ = NewtonSolveType::DECOUPLED_FULL;
    }
  } else {
    if (solver_opts.newton_mode == "algebraic") {
      DEBUG("NewtonSolver: WARNING - algebraic mode requested but mass matrix "
            "has nonzeros in algebraic rows. Falling back to coupled full mode.");
    }
    solve_type_ = NewtonSolveType::COUPLED_FULL;
  }

  DEBUG("NewtonSolver: solve_type = " <<
    (solve_type_ == NewtonSolveType::DECOUPLED_FULL ? "DECOUPLED_FULL" :
     solve_type_ == NewtonSolveType::DECOUPLED_SUBBLOCK ? "DECOUPLED_SUBBLOCK" :
     "COUPLED_FULL"));

  // Pre-allocate buffers
  ewt_.resize(n_states_);
  y_iter_save_.resize(n_states_);
  if (solve_type_ == NewtonSolveType::COUPLED_FULL) {
    yp_iter_save_.resize(n_states_);
    y0_save_.resize(n_states_);
    yp0_save_.resize(n_states_);
  }

  // Allocate scratch buffers for ATimes when using iterative solver
  if (setup_opts_.using_iterative_solver &&
      (solve_type_ == NewtonSolveType::DECOUPLED_FULL ||
       solve_type_ == NewtonSolveType::COUPLED_FULL)) {
    atimes_tmp_.resize(n_states_);
    if (solve_type_ == NewtonSolveType::DECOUPLED_FULL) {
      atimes_v_save_.resize(n_states_);
    }
  }

  // Allocate resources based on solve type
  switch (solve_type_) {
    case NewtonSolveType::DECOUPLED_FULL: {
      res_full_vec_ = N_VNew_Serial(n_states_, sunctx_);
      delta_full_vec_ = N_VNew_Serial(n_states_, sunctx_);
      res_full_buf_.resize(n_states_);
      CopySparsityToJIda();
      break;
    }
    case NewtonSolveType::COUPLED_FULL: {
      res_full_vec_ = N_VNew_Serial(n_states_, sunctx_);
      delta_full_vec_ = N_VNew_Serial(n_states_, sunctx_);
      CopySparsityToJIda();
      break;
    }
    case NewtonSolveType::DECOUPLED_SUBBLOCK: {
      PrecomputeSubBlockSparsity(jac_colptrs, jac_rowvals);

      res_alg_vec_ = N_VNew_Serial(len_alg_, sunctx_);
      delta_alg_vec_ = N_VNew_Serial(len_alg_, sunctx_);

      if (setup_opts_.jacobian == "sparse" &&
          setup_opts_.linear_solver == "SUNLinSol_KLU") {
        J_alg_ = SUNSparseMatrix(len_alg_, len_alg_, alg_nnz_, CSC_MAT, sunctx_);
        LS_alg_ = SUNLinSol_KLU(res_alg_vec_, J_alg_, sunctx_);
      } else {
        J_alg_ = SUNDenseMatrix(len_alg_, len_alg_, sunctx_);
        LS_alg_ = SUNLinSol_Dense(res_alg_vec_, J_alg_, sunctx_);
      }
      SUNLinSolInitialize(LS_alg_);

      // Buffer for jac_alg output (len_alg * n_states or sparse nnz)
      if (functions_->alg_jac) {
        int jac_alg_nnz = functions_->alg_jac->nnz_out();
        full_jac_buf_.resize(jac_alg_nnz > 0 ? jac_alg_nnz : len_alg_ * n_states_);
      } else {
        full_jac_buf_.resize(functions_->number_of_nnz > 0
          ? functions_->number_of_nnz : n_states_ * n_states_);
        res_full_buf_.resize(n_states_);
      }

      CopySparsityToJAlg();
      break;
    }
  }
}

// Destructor

template <class ExprSet>
NewtonSolver<ExprSet>::~NewtonSolver() {
  DEBUG("NewtonSolver::~NewtonSolver");
  if (LS_alg_) SUNLinSolFree(LS_alg_);
  if (J_alg_) SUNMatDestroy(J_alg_);
  if (res_alg_vec_) N_VDestroy(res_alg_vec_);
  if (delta_alg_vec_) N_VDestroy(delta_alg_vec_);
  if (res_full_vec_) N_VDestroy(res_full_vec_);
  if (delta_full_vec_) N_VDestroy(delta_full_vec_);
}

// Mass matrix alignment check

template <class ExprSet>
bool NewtonSolver<ExprSet>::CheckMassMatrixAlignment(const sunrealtype* id_val) {
  std::vector<sunrealtype> e_in(n_states_, 0.0);
  std::vector<sunrealtype> m_out(n_states_, 0.0);

  for (int j = 0; j < n_states_; j++) {
    e_in[j] = 1.0;
    functions_->mass_action->m_arg[0] = e_in.data();
    functions_->mass_action->m_res[0] = m_out.data();
    (*functions_->mass_action)();
    e_in[j] = 0.0;

    for (int i = 0; i < n_states_; i++) {
      if (is_algebraic(id_val[i]) && m_out[i] != 0.0) {
        DEBUG("NewtonSolver: mass matrix has nonzero at algebraic row "
              << i << ", column " << j);
        return false;
      }
    }
  }
  return true;
}

// Sub-block sparsity precomputation

template <class ExprSet>
void NewtonSolver<ExprSet>::PrecomputeSubBlockSparsity(
  const std::vector<int64_t>& jac_colptrs,
  const std::vector<int64_t>& jac_rowvals
) {
  if (functions_->alg_jac) {
    // When we have direct alg_jac, we compute the sub-block sparsity from it
    // alg_jac has shape (len_alg x n_states), we need the (len_alg x len_alg)
    // sub-block corresponding to columns [len_rhs, n_states)
    const auto& rows = functions_->alg_jac->get_row();
    const auto& cols = functions_->alg_jac->get_col();
    int nnz_total = functions_->alg_jac->nnz_out();

    alg_colptrs_.resize(len_alg_ + 1, 0);
    alg_rowvals_.clear();
    alg_data_indices_.clear();

    for (int k = 0; k < nnz_total; k++) {
      int col = static_cast<int>(cols[k]);
      int row = static_cast<int>(rows[k]);
      if (col >= len_rhs_) {
        alg_rowvals_.push_back(row);
        alg_data_indices_.push_back(k);
      }
    }

    // Build CSC colptrs from filtered entries
    // Re-do properly: iterate in column order
    alg_rowvals_.clear();
    alg_data_indices_.clear();
    for (int alg_col = 0; alg_col < len_alg_; alg_col++) {
      alg_colptrs_[alg_col] = static_cast<int64_t>(alg_rowvals_.size());
      int full_col = alg_col + len_rhs_;
      for (int k = 0; k < nnz_total; k++) {
        if (static_cast<int>(cols[k]) == full_col) {
          alg_rowvals_.push_back(rows[k]);
          alg_data_indices_.push_back(k);
        }
      }
    }
    alg_colptrs_[len_alg_] = static_cast<int64_t>(alg_rowvals_.size());
    alg_nnz_ = static_cast<int>(alg_rowvals_.size());
    return;
  }

  // Fallback: extract from full jac_times_cjmass sparsity
  if (jac_colptrs.empty()) {
    alg_nnz_ = len_alg_ * len_alg_;
    return;
  }

  alg_colptrs_.resize(len_alg_ + 1, 0);
  alg_data_indices_.clear();
  alg_rowvals_.clear();

  for (int col = len_rhs_; col < n_states_; col++) {
    int alg_col = col - len_rhs_;
    alg_colptrs_[alg_col] = static_cast<int64_t>(alg_rowvals_.size());
    for (int64_t k = jac_colptrs[col]; k < jac_colptrs[col + 1]; k++) {
      int row = static_cast<int>(jac_rowvals[k]);
      if (row >= len_rhs_) {
        alg_rowvals_.push_back(row - len_rhs_);
        alg_data_indices_.push_back(static_cast<int>(k));
      }
    }
  }
  alg_colptrs_[len_alg_] = static_cast<int64_t>(alg_rowvals_.size());
  alg_nnz_ = static_cast<int>(alg_rowvals_.size());
}

// One-time sparsity pattern initialization

template <class ExprSet>
void NewtonSolver<ExprSet>::CopySparsityToJIda() {
  if (J_ida_ == nullptr) return;
  if (!setup_opts_.using_sparse_matrix) return;
  sunindextype* colptrs = SUNSparseMatrix_IndexPointers(J_ida_);
  sunindextype* rowvals = SUNSparseMatrix_IndexValues(J_ida_);
  const auto& src_colptrs = functions_->jac_times_cjmass_colptrs;
  const auto& src_rowvals = functions_->jac_times_cjmass_rowvals;
  for (size_t i = 0; i < src_colptrs.size(); i++)
    colptrs[i] = src_colptrs[i];
  for (size_t i = 0; i < src_rowvals.size(); i++)
    rowvals[i] = src_rowvals[i];
}

template <class ExprSet>
void NewtonSolver<ExprSet>::CopySparsityToJAlg() {
  bool is_sparse = (setup_opts_.jacobian == "sparse" &&
                    setup_opts_.linear_solver == "SUNLinSol_KLU");
  if (!is_sparse) return;
  sunindextype* colptrs = SUNSparseMatrix_IndexPointers(J_alg_);
  sunindextype* rowvals = SUNSparseMatrix_IndexValues(J_alg_);
  for (int i = 0; i <= len_alg_; i++)
    colptrs[i] = alg_colptrs_[i];
  for (int i = 0; i < alg_nnz_; i++)
    rowvals[i] = alg_rowvals_[i];
}

// WRMS norm over algebraic components of a full-size vector.
// ewt_ must be populated before calling.
template <class ExprSet>
sunrealtype NewtonSolver<ExprSet>::WrmsNormAlg(const sunrealtype* vals) const {
  const sunrealtype* id_data = NV_DATA(id_);
  sunrealtype sum = SUN_RCONST(0.0);
  int count = 0;
  for (int i = 0; i < n_states_; i++) {
    if (is_algebraic(id_data[i])) {
      sunrealtype w = vals[i] * ewt_[i];
      sum += w * w;
      count++;
    }
  }
  return (count > 0) ? std::sqrt(sum / count) : SUN_RCONST(0.0);
}

// WRMS norm over a compact algebraic-only vector of length n.
// ewt_ must be populated with n algebraic weights.
template <class ExprSet>
sunrealtype NewtonSolver<ExprSet>::WrmsNormAlgCompact(
  const sunrealtype* vals, int n
) const {
  sunrealtype sum = SUN_RCONST(0.0);
  for (int i = 0; i < n; i++) {
    sunrealtype w = vals[i] * ewt_[i];
    sum += w * w;
  }
  return (n > 0) ? std::sqrt(sum / n) : SUN_RCONST(0.0);
}

// Jv product: Jv = (dF/dy)*v + cj*(dF/dyp)*v using jac_action + mass_action

template <class ExprSet>
int NewtonSolver<ExprSet>::ComputeJv(N_Vector v, N_Vector Jv) {
  sunrealtype tt = SUN_RCONST(0.0);

  // Jv = (dF/dy) * v
  functions_->jac_action->m_arg[0] = &tt;
  functions_->jac_action->m_arg[1] = NV_DATA(yy_);
  functions_->jac_action->m_arg[2] = functions_->inputs.data();
  functions_->jac_action->m_arg[3] = NV_DATA(v);
  functions_->jac_action->m_res[0] = NV_DATA(Jv);
  (*functions_->jac_action)();

  // tmp = M * v
  functions_->mass_action->m_arg[0] = NV_DATA(v);
  functions_->mass_action->m_res[0] = atimes_tmp_.data();
  (*functions_->mass_action)();

  // Jv -= cj * tmp  =>  Jv = (dF/dy - cj*M) * v  =  (dF/dy + cj*dF/dyp) * v
  axpy(n_states_, -newton_cj_, atimes_tmp_.data(), NV_DATA(Jv));

  return 0;
}

// ATimes for DECOUPLED_FULL: zero differential columns and rows of J, identity on diagonal.
// Equivalent to ZeroDiffRowsColsJacobian but in matrix-free form.
// We temporarily zero the differential components of v so that Jv only picks up
// algebraic-column contributions, then restore v and set differential rows to v[i].
template <class ExprSet>
int NewtonSolver<ExprSet>::newton_atimes_decoupled(void* data, N_Vector v, N_Vector z) {
  auto* self = static_cast<NewtonSolver<ExprSet>*>(data);
  const sunrealtype* id_data = NV_DATA(self->id_);
  sunrealtype* v_data = NV_DATA(v);
  sunrealtype* z_data = NV_DATA(z);

  // Save and zero differential components of v (zeroes differential columns)
  for (int i = 0; i < self->n_states_; i++) {
    if (is_differential(id_data[i])) {
      self->atimes_v_save_[i] = v_data[i];
      v_data[i] = SUN_RCONST(0.0);
    }
  }

  self->ComputeJv(v, z);

  // Restore v and set differential rows: z[i] = v_orig[i] (identity on diagonal)
  for (int i = 0; i < self->n_states_; i++) {
    if (is_differential(id_data[i])) {
      v_data[i] = self->atimes_v_save_[i];
      z_data[i] = self->atimes_v_save_[i];
    }
  }
  return 0;
}

// ATimes for COUPLED_FULL: full Jv, no modification
template <class ExprSet>
int NewtonSolver<ExprSet>::newton_atimes_full(void* data, N_Vector v, N_Vector z) {
  auto* self = static_cast<NewtonSolver<ExprSet>*>(data);
  return self->ComputeJv(v, z);
}

// Register our custom ATimes on the iterative solver
template <class ExprSet>
void NewtonSolver<ExprSet>::SetupATimes() {
  if (!setup_opts_.using_iterative_solver) return;
  SUNATimesFn atimes_fn = (solve_type_ == NewtonSolveType::DECOUPLED_FULL)
    ? &NewtonSolver<ExprSet>::newton_atimes_decoupled
    : &NewtonSolver<ExprSet>::newton_atimes_full;
  SUNLinSolSetATimes(LS_ida_, this, atimes_fn);
}

// Restore IDA's ATimes after Newton solve completes
template <class ExprSet>
void NewtonSolver<ExprSet>::RestoreATimes() {
  if (!setup_opts_.using_iterative_solver) return;
  SUNLinSolSetATimes(LS_ida_, ida_mem_, idaLsATimes);
}

// Dispatch

template <class ExprSet>
NewtonResult NewtonSolver<ExprSet>::solve(
  sunrealtype t, sunrealtype t_next, SolverLog& log
) {
  if (len_alg_ <= 0) return NewtonResult::CONVERGED_WRMS_AND_STEPTOL;

  SetupATimes();

  NewtonResult result;
  switch (solve_type_) {
    case NewtonSolveType::DECOUPLED_FULL:
      result = solve_decoupled_full(t, t_next, log);
      break;
    case NewtonSolveType::DECOUPLED_SUBBLOCK:
      result = solve_decoupled_subblock(t, log);
      break;
    case NewtonSolveType::COUPLED_FULL:
      result = solve_coupled_full(t, t_next, log);
      break;
    default:
      result = NewtonResult::MAX_ITER_NO_CONVERGE;
      break;
  }

  RestoreATimes();

  return result;
}

// Shared helpers

template <class ExprSet>
void NewtonSolver<ExprSet>::EvalRhsAlg(sunrealtype t, sunrealtype* res_out) {
  functions_->rhs_alg->m_arg[0] = &t;
  functions_->rhs_alg->m_arg[1] = N_VGetArrayPointer(yy_);
  functions_->rhs_alg->m_arg[2] = functions_->inputs.data();
  functions_->rhs_alg->m_res[0] = res_out;
  (*functions_->rhs_alg)();
}

template <class ExprSet>
void NewtonSolver<ExprSet>::EvalAlgRes(sunrealtype t, sunrealtype* res_out) {
  functions_->alg_res->m_arg[0] = &t;
  functions_->alg_res->m_arg[1] = N_VGetArrayPointer(yy_);
  functions_->alg_res->m_arg[2] = functions_->inputs.data();
  functions_->alg_res->m_res[0] = res_out;
  (*functions_->alg_res)();
}

template <class ExprSet>
void NewtonSolver<ExprSet>::EvalAlgJac(sunrealtype t, sunrealtype* jac_out) {
  functions_->alg_jac->m_arg[0] = &t;
  functions_->alg_jac->m_arg[1] = N_VGetArrayPointer(yy_);
  functions_->alg_jac->m_arg[2] = functions_->inputs.data();
  functions_->alg_jac->m_res[0] = jac_out;
  (*functions_->alg_jac)();
}

template <class ExprSet>
void NewtonSolver<ExprSet>::EvalFullResidual(sunrealtype t) {
  residual_eval<ExprSet>(t, yy_, yyp_, res_full_vec_, functions_);
}

template <class ExprSet>
void NewtonSolver<ExprSet>::EvalFullJacobian(sunrealtype t, sunrealtype cj) {
  sunrealtype* jac_data;
  if (setup_opts_.using_sparse_matrix) {
    jac_data = SUNSparseMatrix_Data(J_ida_);
  } else if (setup_opts_.using_banded_matrix) {
    jac_data = functions_->get_tmp_sparse_jacobian_data();
  } else {
    jac_data = SUNDenseMatrix_Data(J_ida_);
  }

  functions_->jac_times_cjmass->m_arg[0] = &t;
  functions_->jac_times_cjmass->m_arg[1] = N_VGetArrayPointer(yy_);
  functions_->jac_times_cjmass->m_arg[2] = functions_->inputs.data();
  functions_->jac_times_cjmass->m_arg[3] = &cj;
  functions_->jac_times_cjmass->m_res[0] = jac_data;
  (*functions_->jac_times_cjmass)();

  if (setup_opts_.using_banded_matrix) {
    auto jac_colptrs_data = functions_->jac_times_cjmass_colptrs.data();
    auto jac_rowvals_data = functions_->jac_times_cjmass_rowvals.data();
    for (int col = 0; col < n_states_; col++) {
      sunrealtype* banded_col = SM_COLUMN_B(J_ida_, col);
      for (auto di = jac_colptrs_data[col]; di < jac_colptrs_data[col+1]; di++) {
        auto row = jac_rowvals_data[di];
        SM_COLUMN_ELEMENT_B(banded_col, row, col) = jac_data[di];
      }
    }
  }
}

// DECOUPLED_FULL: zero diff rows/cols, identity on diff diagonal, cj=1

template <class ExprSet>
void NewtonSolver<ExprSet>::ZeroDiffRowsColsJacobian() {
  const sunrealtype* id_data = N_VGetArrayPointer(id_);

  if (setup_opts_.using_sparse_matrix) {
    sunindextype* colptrs = SUNSparseMatrix_IndexPointers(J_ida_);
    sunindextype* rowvals = SUNSparseMatrix_IndexValues(J_ida_);
    sunrealtype* data = SUNSparseMatrix_Data(J_ida_);

    for (int col = 0; col < n_states_; col++) {
      bool diff_col = is_differential(id_data[col]);
      for (sunindextype k = colptrs[col]; k < colptrs[col + 1]; k++) {
        int row = static_cast<int>(rowvals[k]);
        bool diff_row = is_differential(id_data[row]);
        if (diff_row || diff_col) {
          data[k] = (row == col) ? SUN_RCONST(1.0) : SUN_RCONST(0.0);
        }
      }
    }
  } else if (setup_opts_.using_banded_matrix) {
    for (int col = 0; col < n_states_; col++) {
      bool diff_col = is_differential(id_data[col]);
      sunrealtype* banded_col = SM_COLUMN_B(J_ida_, col);
      for (int row = 0; row < n_states_; row++) {
        bool diff_row = is_differential(id_data[row]);
        if (diff_row || diff_col) {
          SM_COLUMN_ELEMENT_B(banded_col, row, col) =
            (row == col) ? SUN_RCONST(1.0) : SUN_RCONST(0.0);
        }
      }
    }
  } else {
    sunrealtype* data = SUNDenseMatrix_Data(J_ida_);
    for (int col = 0; col < n_states_; col++) {
      bool diff_col = is_differential(id_data[col]);
      for (int row = 0; row < n_states_; row++) {
        bool diff_row = is_differential(id_data[row]);
        if (diff_row || diff_col) {
          data[col * n_states_ + row] =
            (row == col) ? SUN_RCONST(1.0) : SUN_RCONST(0.0);
        }
      }
    }
  }
}

// DECOUPLED_FULL: zero differential residuals, reuse IDA's LS, cj=1

template <class ExprSet>
NewtonResult NewtonSolver<ExprSet>::solve_decoupled_full(
  sunrealtype t, sunrealtype t_next, SolverLog& log
) {
  log.log_newton_start(t, len_alg_);

  const int max_iter = solver_opts_.max_num_iterations_ic;
  const sunrealtype epsNewt = solver_opts_.nonlinear_convergence_coefficient_ic;
  const sunrealtype abstolStep = solver_opts_.newton_step_tol;

  sunrealtype* y_val = N_VGetArrayPointer(yy_);
  sunrealtype* res_data = N_VGetArrayPointer(res_full_vec_);
  sunrealtype* delta_data = N_VGetArrayPointer(delta_full_vec_);
  const sunrealtype* id_data = N_VGetArrayPointer(id_);
  const sunrealtype* atol_data = N_VGetArrayPointer(avtol_);

  sunrealtype delnorm = std::numeric_limits<sunrealtype>::infinity();
  sunrealtype prev_res_norm = std::numeric_limits<sunrealtype>::infinity();
  bool converged = false;

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute ewt: ewt[i] = 1 / (rtol * |y[i]| + atol[i])
    for (int i = 0; i < n_states_; i++) {
      ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(y_val[i]) + atol_data[i]);
    }

    EvalRhsAlg(t, res_full_buf_.data());
    for (int i = 0; i < n_states_; i++) {
      res_data[i] = is_differential(id_data[i]) ? SUN_RCONST(0.0) : res_full_buf_[i];
    }

    sunrealtype res_norm = WrmsNormAlg(res_full_buf_.data());
    log.log_newton_iteration(iter, res_norm, delnorm);

    newton_cj_ = SUN_RCONST(1.0);
    int flag = 0;
    if (!setup_opts_.using_iterative_solver) {
      EvalFullJacobian(t, newton_cj_);
      ZeroDiffRowsColsJacobian();
      flag = SUNLinSolSetup(LS_ida_, J_ida_);
    }
    if (flag != 0) {
      log.log_newton_failed(iter + 1, res_norm, newton_result_reason(NewtonResult::LSETUP_FAIL));
      return NewtonResult::LSETUP_FAIL;
    }

    flag = SUNLinSolSolve(LS_ida_, J_ida_, delta_full_vec_, res_full_vec_,
                          SUN_RCONST(0.0));
    if (flag != 0) {
      log.log_newton_failed(iter + 1, res_norm, newton_result_reason(NewtonResult::LSOLVE_FAIL));
      return NewtonResult::LSOLVE_FAIL;
    }

    delnorm = WrmsNormAlg(delta_data);

    // Convergence check: WRMS(delta, ewt) <= epsNewt
    if (delnorm <= epsNewt) {
      converged = true;
      if (delnorm <= abstolStep) {
        log.log_newton_converged(iter + 1, newton_result_reason(NewtonResult::CONVERGED_WRMS_AND_STEPTOL));
        for (int i = 0; i < n_states_; i++) {
          if (is_algebraic(id_data[i])) y_val[i] -= delta_data[i];
        }
        return NewtonResult::CONVERGED_WRMS_AND_STEPTOL;
      }
      if (iter > 0 && res_norm >= prev_res_norm) {
        for (int i = 0; i < n_states_; i++) {
          if (is_algebraic(id_data[i])) y_val[i] = y_iter_save_[i];
        }
        log.log_newton_converged(iter + 1, newton_result_reason(NewtonResult::CONVERGED_WRMS_STEP_DIVERGED));
        return NewtonResult::CONVERGED_WRMS_STEP_DIVERGED;
      }
    }

    prev_res_norm = res_norm;
    for (int i = 0; i < n_states_; i++) {
      if (is_algebraic(id_data[i])) {
        y_iter_save_[i] = y_val[i];
      }
    }

    // Line search with Armijo condition on WRMS norm of residual
    sunrealtype alpha = SUN_RCONST(1.0);
    while (true) {
      for (int i = 0; i < n_states_; i++) {
        if (is_algebraic(id_data[i])) {
          y_val[i] = y_iter_save_[i] - alpha * delta_data[i];
        }
      }

      EvalRhsAlg(t, res_full_buf_.data());
      sunrealtype trial_norm = WrmsNormAlg(res_full_buf_.data());

      if (trial_norm <= (SUN_RCONST(1.0) - alpha * SUN_RCONST(0.5)) * res_norm) {
        break;
      }
      if (alpha * delnorm <= abstolStep) {
        break;
      }
      alpha *= SUN_RCONST(0.5);
    }
  }

  if (converged) {
    log.log_newton_converged(max_iter, newton_result_reason(NewtonResult::CONVERGED_WRMS_AT_MAX_ITER));
    return NewtonResult::CONVERGED_WRMS_AT_MAX_ITER;
  }
  for (int i = 0; i < n_states_; i++) {
    ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(y_val[i]) + atol_data[i]);
  }
  EvalRhsAlg(t, res_full_buf_.data());
  sunrealtype final_norm = WrmsNormAlg(res_full_buf_.data());
  log.log_newton_failed(max_iter, final_norm, newton_result_reason(NewtonResult::MAX_ITER_NO_CONVERGE));
  return NewtonResult::MAX_ITER_NO_CONVERGE;
}

// DECOUPLED_SUBBLOCK: n_alg x n_alg with own linear solver

template <class ExprSet>
NewtonResult NewtonSolver<ExprSet>::solve_decoupled_subblock(
  sunrealtype t, SolverLog& log
) {
  log.log_newton_start(t, len_alg_);

  const int max_iter = solver_opts_.max_num_iterations_ic;
  const sunrealtype epsNewt = solver_opts_.nonlinear_convergence_coefficient_ic;
  const sunrealtype abstolStep = solver_opts_.newton_step_tol;

  sunrealtype* y_val = N_VGetArrayPointer(yy_);
  sunrealtype* y_alg = y_val + len_rhs_;
  sunrealtype* res_data = N_VGetArrayPointer(res_alg_vec_);
  sunrealtype* delta_data = N_VGetArrayPointer(delta_alg_vec_);
  const sunrealtype* atol_data = N_VGetArrayPointer(avtol_);

  sunrealtype delnorm = std::numeric_limits<sunrealtype>::infinity();
  sunrealtype prev_res_norm = std::numeric_limits<sunrealtype>::infinity();
  bool converged = false;

  bool use_direct_alg = (functions_->alg_res != nullptr &&
                         functions_->alg_jac != nullptr);

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute ewt for algebraic variables: ewt[i] = 1 / (rtol * |y_alg[i]| + atol[i])
    for (int i = 0; i < len_alg_; i++) {
      ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(y_alg[i]) + atol_data[len_rhs_ + i]);
    }

    if (use_direct_alg) {
      EvalAlgRes(t, res_data);
    } else {
      EvalRhsAlg(t, res_full_buf_.data());
      std::memcpy(res_data, res_full_buf_.data() + len_rhs_,
                  len_alg_ * sizeof(sunrealtype));
    }

    sunrealtype res_norm = WrmsNormAlgCompact(res_data, len_alg_);
    log.log_newton_iteration(iter, res_norm, delnorm);

    if (use_direct_alg) {
      EvalAlgJac(t, full_jac_buf_.data());

      bool is_sparse = (setup_opts_.jacobian == "sparse" &&
                        setup_opts_.linear_solver == "SUNLinSol_KLU");
      if (is_sparse) {
        sunrealtype* alg_data = SUNSparseMatrix_Data(J_alg_);
        for (int i = 0; i < alg_nnz_; i++) {
          alg_data[i] = full_jac_buf_[alg_data_indices_[i]];
        }
      } else {
        sunrealtype* alg_data = SUNDenseMatrix_Data(J_alg_);
        for (int j = 0; j < len_alg_; j++) {
          int full_col = j + len_rhs_;
          for (int i = 0; i < len_alg_; i++) {
            alg_data[j * len_alg_ + i] =
              full_jac_buf_[full_col * len_alg_ + i];
          }
        }
      }
    } else {
      sunrealtype cj_zero = SUN_RCONST(0.0);
      functions_->jac_times_cjmass->m_arg[0] = &t;
      functions_->jac_times_cjmass->m_arg[1] = N_VGetArrayPointer(yy_);
      functions_->jac_times_cjmass->m_arg[2] = functions_->inputs.data();
      functions_->jac_times_cjmass->m_arg[3] = &cj_zero;
      functions_->jac_times_cjmass->m_res[0] = full_jac_buf_.data();
      (*functions_->jac_times_cjmass)();

      bool is_sparse = (setup_opts_.jacobian == "sparse" &&
                        setup_opts_.linear_solver == "SUNLinSol_KLU");
      if (is_sparse) {
        sunrealtype* alg_data = SUNSparseMatrix_Data(J_alg_);
        for (int i = 0; i < alg_nnz_; i++) {
          alg_data[i] = full_jac_buf_[alg_data_indices_[i]];
        }
      } else {
        sunrealtype* alg_data = SUNDenseMatrix_Data(J_alg_);
        for (int j = 0; j < len_alg_; j++) {
          for (int i = 0; i < len_alg_; i++) {
            int full_row = i + len_rhs_;
            int full_col = j + len_rhs_;
            alg_data[j * len_alg_ + i] =
              full_jac_buf_[full_col * n_states_ + full_row];
          }
        }
      }
    }

    int flag = SUNLinSolSetup(LS_alg_, J_alg_);
    if (flag != 0) {
      log.log_newton_failed(iter + 1, res_norm, newton_result_reason(NewtonResult::LSETUP_FAIL));
      return NewtonResult::LSETUP_FAIL;
    }

    flag = SUNLinSolSolve(LS_alg_, J_alg_, delta_alg_vec_, res_alg_vec_,
                          SUN_RCONST(0.0));
    if (flag != 0) {
      log.log_newton_failed(iter + 1, res_norm, newton_result_reason(NewtonResult::LSOLVE_FAIL));
      return NewtonResult::LSOLVE_FAIL;
    }

    delnorm = WrmsNormAlgCompact(delta_data, len_alg_);

    // Convergence check: WRMS(delta, ewt) <= epsNewt
    if (delnorm <= epsNewt) {
      converged = true;
      if (delnorm <= abstolStep) {
        for (int i = 0; i < len_alg_; i++) y_alg[i] -= delta_data[i];
        log.log_newton_converged(iter + 1, newton_result_reason(NewtonResult::CONVERGED_WRMS_AND_STEPTOL));
        return NewtonResult::CONVERGED_WRMS_AND_STEPTOL;
      }
      if (iter > 0 && res_norm >= prev_res_norm) {
        std::memcpy(y_alg, y_iter_save_.data(),
                    len_alg_ * sizeof(sunrealtype));
        log.log_newton_converged(iter + 1, newton_result_reason(NewtonResult::CONVERGED_WRMS_STEP_DIVERGED));
        return NewtonResult::CONVERGED_WRMS_STEP_DIVERGED;
      }
    }

    prev_res_norm = res_norm;
    std::memcpy(y_iter_save_.data(), y_alg,
                len_alg_ * sizeof(sunrealtype));

    // Line search with Armijo condition on WRMS norm of residual
    sunrealtype alpha = SUN_RCONST(1.0);
    while (true) {
      for (int i = 0; i < len_alg_; i++) {
        y_alg[i] = y_iter_save_[i] - alpha * delta_data[i];
      }

      sunrealtype trial_norm;
      if (use_direct_alg) {
        EvalAlgRes(t, res_data);
        trial_norm = WrmsNormAlgCompact(res_data, len_alg_);
      } else {
        EvalRhsAlg(t, res_full_buf_.data());
        trial_norm = WrmsNormAlgCompact(res_full_buf_.data() + len_rhs_, len_alg_);
      }

      if (trial_norm <= (SUN_RCONST(1.0) - alpha * SUN_RCONST(0.5)) * res_norm) {
        break;
      }
      if (alpha * delnorm <= abstolStep) {
        break;
      }
      alpha *= SUN_RCONST(0.5);
    }
  }

  if (converged) {
    log.log_newton_converged(max_iter, newton_result_reason(NewtonResult::CONVERGED_WRMS_AT_MAX_ITER));
    return NewtonResult::CONVERGED_WRMS_AT_MAX_ITER;
  }
  for (int i = 0; i < len_alg_; i++) {
    ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(y_alg[i]) + atol_data[len_rhs_ + i]);
  }
  sunrealtype final_norm;
  if (use_direct_alg) {
    EvalAlgRes(t, res_data);
    final_norm = WrmsNormAlgCompact(res_data, len_alg_);
  } else {
    EvalRhsAlg(t, res_full_buf_.data());
    final_norm = WrmsNormAlgCompact(res_full_buf_.data() + len_rhs_, len_alg_);
  }
  log.log_newton_failed(max_iter, final_norm, newton_result_reason(NewtonResult::MAX_ITER_NO_CONVERGE));
  return NewtonResult::MAX_ITER_NO_CONVERGE;
}

// COUPLED_FULL: IDACalcIC-style solve for y_alg and ydot_diff

template <class ExprSet>
NewtonResult NewtonSolver<ExprSet>::solve_coupled_full(
  sunrealtype t, sunrealtype t_next, SolverLog& log
) {
  log.log_newton_start(t, len_alg_);

  const int max_iter = solver_opts_.max_num_iterations_ic;
  const sunrealtype epsNewt = solver_opts_.nonlinear_convergence_coefficient_ic;
  const sunrealtype abstolStep = solver_opts_.newton_step_tol;
  const int max_nh = solver_opts_.max_num_steps_ic > 0;

  sunrealtype* y_val = N_VGetArrayPointer(yy_);
  sunrealtype* yp_val = N_VGetArrayPointer(yyp_);
  sunrealtype* res_data = N_VGetArrayPointer(res_full_vec_);
  sunrealtype* delta_data = N_VGetArrayPointer(delta_full_vec_);
  const sunrealtype* id_data = N_VGetArrayPointer(id_);
  const sunrealtype* atol_data = N_VGetArrayPointer(avtol_);

  sunrealtype hic = SUN_RCONST(0.001) * std::abs(t_next - t);
  if (hic == SUN_RCONST(0.0)) hic = SUN_RCONST(1.0e-6);

  std::memcpy(y0_save_.data(), y_val, n_states_ * sizeof(sunrealtype));
  std::memcpy(yp0_save_.data(), yp_val, n_states_ * sizeof(sunrealtype));

  for (int nh = 0; nh < max_nh; nh++) {
    sunrealtype cj = SUN_RCONST(1.0) / hic;
    sunrealtype delnorm = std::numeric_limits<sunrealtype>::infinity();
    sunrealtype prev_res_norm = std::numeric_limits<sunrealtype>::infinity();
    NewtonResult inner_result = NewtonResult::MAX_ITER_NO_CONVERGE;

    if (nh > 0) {
      std::memcpy(y_val, y0_save_.data(), n_states_ * sizeof(sunrealtype));
      std::memcpy(yp_val, yp0_save_.data(), n_states_ * sizeof(sunrealtype));
    }

    for (int iter = 0; iter < max_iter; iter++) {
      // Compute ewt: ewt[i] = 1 / (rtol * |y[i]| + atol[i])
      for (int i = 0; i < n_states_; i++) {
        ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(y_val[i]) + atol_data[i]);
      }

      EvalFullResidual(t);
      sunrealtype res_norm = WrmsNormAlg(res_data);
      log.log_newton_iteration(iter, res_norm, delnorm);

      newton_cj_ = cj;
      int flag = 0;
      if (!setup_opts_.using_iterative_solver) {
        EvalFullJacobian(t, cj);
        flag = SUNLinSolSetup(LS_ida_, J_ida_);
      }
      if (flag != 0) break;

      flag = SUNLinSolSolve(LS_ida_, J_ida_, delta_full_vec_, res_full_vec_,
                            SUN_RCONST(0.0));
      if (flag != 0) break;

      delnorm = WrmsNormAlg(delta_data);

      // Convergence check: WRMS(delta, ewt) <= epsNewt
      if (delnorm <= epsNewt) {
        if (delnorm <= abstolStep) {
          for (int i = 0; i < n_states_; i++) {
            if (is_algebraic(id_data[i])) {
              y_val[i] -= delta_data[i];
            } else {
              yp_val[i] -= cj * delta_data[i];
            }
          }
          inner_result = NewtonResult::CONVERGED_WRMS_AND_STEPTOL;
          log.log_newton_converged(iter + 1, newton_result_reason(inner_result));
          break;
        }
        if (iter > 0 && res_norm >= prev_res_norm) {
          for (int i = 0; i < n_states_; i++) {
            if (is_algebraic(id_data[i])) {
              y_val[i] = y_iter_save_[i];
            } else {
              yp_val[i] = yp_iter_save_[i];
            }
          }
          inner_result = NewtonResult::CONVERGED_WRMS_STEP_DIVERGED;
          log.log_newton_converged(iter + 1, newton_result_reason(inner_result));
          break;
        }
        inner_result = NewtonResult::CONVERGED_WRMS_AT_MAX_ITER;
      }

      prev_res_norm = res_norm;
      for (int i = 0; i < n_states_; i++) {
        if (is_algebraic(id_data[i])) {
          y_iter_save_[i] = y_val[i];
        } else {
          yp_iter_save_[i] = yp_val[i];
        }
      }

      // Line search with Armijo condition on WRMS norm of residual
      sunrealtype alpha = SUN_RCONST(1.0);
      while (true) {
        for (int i = 0; i < n_states_; i++) {
          if (is_algebraic(id_data[i])) {
            y_val[i] = y_iter_save_[i] - alpha * delta_data[i];
          } else {
            yp_val[i] = yp_iter_save_[i] - cj * alpha * delta_data[i];
          }
        }

        EvalFullResidual(t);
        sunrealtype trial_norm = WrmsNormAlg(res_data);

        if (trial_norm <= (SUN_RCONST(1.0) - alpha * SUN_RCONST(0.5)) * res_norm) {
          break;
        }
        if (alpha * delnorm <= abstolStep) {
          break;
        }
        alpha *= SUN_RCONST(0.5);
      }
    }

    if (newton_success(inner_result)) return inner_result;

    hic *= SUN_RCONST(0.1);
  }

  for (int i = 0; i < n_states_; i++) {
    ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(y_val[i]) + atol_data[i]);
  }
  EvalFullResidual(t);
  sunrealtype final_norm = WrmsNormAlg(res_data);
  log.log_newton_failed(max_iter, final_norm, newton_result_reason(NewtonResult::MAX_ITER_NO_CONVERGE));
  return NewtonResult::MAX_ITER_NO_CONVERGE;
}
