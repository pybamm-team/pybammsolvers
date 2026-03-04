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
    atol_data_(N_VGetArrayPointer(avtol)),
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
    delta_full_vec_(nullptr),
    res_data_(nullptr),
    delta_data_(nullptr)
{
  DEBUG("NewtonSolver::NewtonSolver (len_rhs=" << len_rhs_ << ", len_alg=" << len_alg_ << ")");

  if (len_alg_ <= 0) {
    solve_type_ = NewtonSolveType::DECOUPLED_FULL;
    return;
  }

  // Cache algebraic/differential index sets from id_val (once, never scanned again)
  for (int i = 0; i < n_states_; i++) {
    if (is_algebraic(id_val[i]))
      alg_idx_.push_back(i);
    else
      diff_idx_.push_back(i);
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
    case NewtonSolveType::DECOUPLED_FULL:
    case NewtonSolveType::COUPLED_FULL: {
      res_full_vec_ = N_VNew_Serial(n_states_, sunctx_);
      delta_full_vec_ = N_VNew_Serial(n_states_, sunctx_);
      CopySparsityToJIda();
      break;
    }
    case NewtonSolveType::DECOUPLED_SUBBLOCK: {
      if (functions_->alg_res == nullptr || functions_->alg_jac == nullptr) {
        throw std::runtime_error(
          "NewtonSolver: algebraic mode requires alg_res and alg_jac functions");
      }

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

      int jac_alg_nnz = functions_->alg_jac->nnz_out();
      full_jac_buf_.resize(jac_alg_nnz > 0 ? jac_alg_nnz : len_alg_ * n_states_);

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
  const auto& rows = functions_->alg_jac->get_row();
  const auto& cols = functions_->alg_jac->get_col();
  int nnz_total = functions_->alg_jac->nnz_out();

  alg_colptrs_.resize(len_alg_ + 1, 0);
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
}

// Sparsity pattern initialization

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

// Helpers

// Compute error weight vector once before the Newton loop (matches IDA behavior).
// SUBBLOCK stores ewt compactly at indices 0..len_alg_-1.
// FULL modes store ewt at global indices via alg_idx_.
template <class ExprSet>
void NewtonSolver<ExprSet>::ComputeEwt() {
  const sunrealtype* y_val = NV_DATA(yy_);
  if (solve_type_ == NewtonSolveType::DECOUPLED_SUBBLOCK) {
    for (int i = 0; i < len_alg_; i++) {
      int gi = alg_idx_[i];
      ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(y_val[gi]) + atol_data_[gi]);
    }
  } else {
    for (int idx : alg_idx_) {
      ewt_[idx] = SUN_RCONST(1.0) / (rtol_ * std::abs(y_val[idx]) + atol_data_[idx]);
    }
  }
}

// WRMS norm over algebraic components (used for step convergence criterion).
// SUBBLOCK: vals is compact (length len_alg_), ewt stored at 0..len_alg_-1.
// FULL modes: vals is full-size, iterate alg_idx_ with ewt at global indices.
template <class ExprSet>
sunrealtype NewtonSolver<ExprSet>::WrmsNorm(const sunrealtype* vals) const {
  const int n = static_cast<int>(alg_idx_.size());
  sunrealtype sum = SUN_RCONST(0.0);
  if (solve_type_ == NewtonSolveType::DECOUPLED_SUBBLOCK) {
    for (int i = 0; i < n; i++) {
      sunrealtype w = vals[i] * ewt_[i];
      sum += w * w;
    }
  } else {
    for (int idx : alg_idx_) {
      sunrealtype w = vals[idx] * ewt_[idx];
      sum += w * w;
    }
  }
  return (n > 0) ? std::sqrt(sum / n) : SUN_RCONST(0.0);
}

// Infinity norm over algebraic components (used for line search / residual checks).
template <class ExprSet>
sunrealtype NewtonSolver<ExprSet>::InfNorm(const sunrealtype* vals) const {
  sunrealtype mx = SUN_RCONST(0.0);
  if (solve_type_ == NewtonSolveType::DECOUPLED_SUBBLOCK) {
    for (int i = 0; i < len_alg_; i++) {
      sunrealtype a = std::abs(vals[i]);
      if (a > mx) mx = a;
    }
  } else {
    for (int idx : alg_idx_) {
      sunrealtype a = std::abs(vals[idx]);
      if (a > mx) mx = a;
    }
  }
  return mx;
}

// Save current iterate.
// SUBBLOCK: saves y_alg (compact, len_alg_ values starting at y_val + len_rhs_).
// FULL modes: saves y at alg_idx_, and yp at diff_idx_ for coupled.
template <class ExprSet>
void NewtonSolver<ExprSet>::SaveIterate() {
  const sunrealtype* y_val = NV_DATA(yy_);
  if (solve_type_ == NewtonSolveType::DECOUPLED_SUBBLOCK) {
    std::memcpy(y_iter_save_.data(), y_val + len_rhs_,
                len_alg_ * sizeof(sunrealtype));
  } else {
    for (int idx : alg_idx_) {
      y_iter_save_[idx] = y_val[idx];
    }
    if (solve_type_ == NewtonSolveType::COUPLED_FULL) {
      const sunrealtype* yp_val = NV_DATA(yyp_);
      for (int idx : diff_idx_) {
        yp_iter_save_[idx] = yp_val[idx];
      }
    }
  }
}

// Revert saved iterate and apply Newton step in a single pass.
// alpha=0 is a pure revert; alpha=1 is a full step from saved state.
template <class ExprSet>
void NewtonSolver<ExprSet>::RevertAndApply(sunrealtype alpha, sunrealtype cj) {
  sunrealtype* y_val = NV_DATA(yy_);
  if (solve_type_ == NewtonSolveType::DECOUPLED_SUBBLOCK) {
    sunrealtype* y_alg = y_val + len_rhs_;
    for (int i = 0; i < len_alg_; i++) {
      y_alg[i] = y_iter_save_[i] - alpha * delta_data_[i];
    }
  } else {
    for (int idx : alg_idx_) {
      y_val[idx] = y_iter_save_[idx] - alpha * delta_data_[idx];
    }
    if (solve_type_ == NewtonSolveType::COUPLED_FULL) {
      sunrealtype* yp_val = NV_DATA(yyp_);
      for (int idx : diff_idx_) {
        yp_val[idx] = yp_iter_save_[idx] - cj * alpha * delta_data_[idx];
      }
    }
  }
}

// Residual/Jacobian evaluation (low-level)

template <class ExprSet>
void NewtonSolver<ExprSet>::EvalRhsAlg(sunrealtype t, sunrealtype* res_out) {
  functions_->rhs_alg->m_arg[0] = &t;
  functions_->rhs_alg->m_arg[1] = NV_DATA(yy_);
  functions_->rhs_alg->m_arg[2] = functions_->inputs.data();
  functions_->rhs_alg->m_res[0] = res_out;
  (*functions_->rhs_alg)();
}

template <class ExprSet>
void NewtonSolver<ExprSet>::EvalAlgRes(sunrealtype t, sunrealtype* res_out) {
  functions_->alg_res->m_arg[0] = &t;
  functions_->alg_res->m_arg[1] = NV_DATA(yy_);
  functions_->alg_res->m_arg[2] = functions_->inputs.data();
  functions_->alg_res->m_res[0] = res_out;
  (*functions_->alg_res)();
}

template <class ExprSet>
void NewtonSolver<ExprSet>::EvalAlgJac(sunrealtype t, sunrealtype* jac_out) {
  functions_->alg_jac->m_arg[0] = &t;
  functions_->alg_jac->m_arg[1] = NV_DATA(yy_);
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
  functions_->jac_times_cjmass->m_arg[1] = NV_DATA(yy_);
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

// DECOUPLED_FULL: zero diff rows/cols, identity on diff diagonal
template <class ExprSet>
void NewtonSolver<ExprSet>::ZeroDiffRowsColsJacobian() {
  const sunrealtype* id_data = NV_DATA(id_);

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

// DECOUPLED_SUBBLOCK: fill J_alg_ from alg_jac
template <class ExprSet>
void NewtonSolver<ExprSet>::FillSubBlockJacobian(sunrealtype t) {
  EvalAlgJac(t, full_jac_buf_.data());

  if (setup_opts_.jacobian == "sparse" &&
      setup_opts_.linear_solver == "SUNLinSol_KLU") {
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
}

// ATimes callbacks for iterative solvers

// Jv = (dF/dy)*v + cj*(dF/dyp)*v = (dF/dy - cj*M)*v
template <class ExprSet>
int NewtonSolver<ExprSet>::ComputeJv(N_Vector v, N_Vector Jv) {
  sunrealtype tt = SUN_RCONST(0.0);

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

  // Jv -= cj * tmp  =>  Jv = (dF/dy - cj*M) * v
  axpy(n_states_, -newton_cj_, atimes_tmp_.data(), NV_DATA(Jv));

  return 0;
}

// DECOUPLED_FULL ATimes: zero differential columns and rows of J, identity on diagonal.
// Temporarily zeros differential components of v so that Jv only picks up
// algebraic-column contributions, then restores v and sets differential rows to v[i].
template <class ExprSet>
int NewtonSolver<ExprSet>::newton_atimes_decoupled(void* data, N_Vector v, N_Vector z) {
  auto* self = static_cast<NewtonSolver<ExprSet>*>(data);
  sunrealtype* v_data = NV_DATA(v);
  sunrealtype* z_data = NV_DATA(z);

  for (int idx : self->diff_idx_) {
    self->atimes_v_save_[idx] = v_data[idx];
    v_data[idx] = SUN_RCONST(0.0);
  }

  self->ComputeJv(v, z);

  for (int idx : self->diff_idx_) {
    v_data[idx] = self->atimes_v_save_[idx];
    z_data[idx] = self->atimes_v_save_[idx];
  }
  return 0;
}

// COUPLED_FULL ATimes: full Jv, no modification
template <class ExprSet>
int NewtonSolver<ExprSet>::newton_atimes_full(void* data, N_Vector v, N_Vector z) {
  auto* self = static_cast<NewtonSolver<ExprSet>*>(data);
  return self->ComputeJv(v, z);
}

template <class ExprSet>
void NewtonSolver<ExprSet>::SetupATimes() {
  if (!setup_opts_.using_iterative_solver) return;
  SUNATimesFn atimes_fn = (solve_type_ == NewtonSolveType::DECOUPLED_FULL)
    ? &NewtonSolver<ExprSet>::newton_atimes_decoupled
    : &NewtonSolver<ExprSet>::newton_atimes_full;
  SUNLinSolSetATimes(LS_ida_, this, atimes_fn);
}

template <class ExprSet>
void NewtonSolver<ExprSet>::RestoreATimes() {
  if (!setup_opts_.using_iterative_solver) return;
  SUNLinSolSetATimes(LS_ida_, ida_mem_, idaLsATimes);
}

// Mode-dispatching evaluation helpers

// Evaluate residual and return its infinity norm (matching CasADi's convention
// for the line search and residual divergence checks).
template <class ExprSet>
sunrealtype NewtonSolver<ExprSet>::EvalResidualAndNorm(sunrealtype t) {
  switch (solve_type_) {
    case NewtonSolveType::DECOUPLED_FULL: {
      sunrealtype* res = NV_DATA(res_full_vec_);
      EvalRhsAlg(t, res);
      for (int idx : diff_idx_) res[idx] = SUN_RCONST(0.0);
      return InfNorm(res);
    }
    case NewtonSolveType::DECOUPLED_SUBBLOCK: {
      EvalAlgRes(t, res_data_);
      return InfNorm(res_data_);
    }
    case NewtonSolveType::COUPLED_FULL: {
      EvalFullResidual(t);
      return InfNorm(NV_DATA(res_full_vec_));
    }
  }
  return SUN_RCONST(0.0);
}

// Setup Jacobian and solve the linear system. Returns 0 on success,
// positive for LSETUP fail, negative for LSOLVE fail.
template <class ExprSet>
int NewtonSolver<ExprSet>::SetupAndSolveLinearSystem(sunrealtype t, sunrealtype cj) {
  int flag = 0;

  if (solve_type_ == NewtonSolveType::DECOUPLED_SUBBLOCK) {
    FillSubBlockJacobian(t);
    flag = SUNLinSolSetup(LS_alg_, J_alg_);
    if (flag != 0) return 1;
    flag = SUNLinSolSolve(LS_alg_, J_alg_, delta_alg_vec_, res_alg_vec_,
                          SUN_RCONST(0.0));
    return (flag != 0) ? -1 : 0;
  }

  // DECOUPLED_FULL and COUPLED_FULL share the same linear solve path
  newton_cj_ = (solve_type_ == NewtonSolveType::DECOUPLED_FULL)
    ? SUN_RCONST(1.0) : cj;
  if (!setup_opts_.using_iterative_solver) {
    EvalFullJacobian(t, newton_cj_);
    if (solve_type_ == NewtonSolveType::DECOUPLED_FULL)
      ZeroDiffRowsColsJacobian();
    flag = SUNLinSolSetup(LS_ida_, J_ida_);
    if (flag != 0) return 1;
  }
  flag = SUNLinSolSolve(LS_ida_, J_ida_, delta_full_vec_, res_full_vec_,
                        SUN_RCONST(0.0));
  return (flag != 0) ? -1 : 0;
}

// Unified Newton iteration loop

template <class ExprSet>
NewtonResult NewtonSolver<ExprSet>::RunNewtonLoop(
  sunrealtype t, sunrealtype cj, SolverLog& log
) {
  const int max_iter = solver_opts_.max_num_iterations_ic;
  const sunrealtype epsNewt = solver_opts_.nonlinear_convergence_coefficient_ic;
  const sunrealtype abstolStep = solver_opts_.newton_step_tol;

  sunrealtype delnorm = std::numeric_limits<sunrealtype>::infinity();
  sunrealtype prev_res_norm = std::numeric_limits<sunrealtype>::infinity();
  bool converged = false;

  ComputeEwt();

  for (int iter = 0; iter < max_iter; iter++) {
    sunrealtype res_norm = EvalResidualAndNorm(t);
    log.log_newton_iteration(iter, res_norm, delnorm);

    int lsflag = SetupAndSolveLinearSystem(t, cj);
    if (lsflag > 0) {
      log.log_newton_failed(iter + 1, res_norm, newton_result_reason(NewtonResult::LSETUP_FAIL));
      return NewtonResult::LSETUP_FAIL;
    }
    if (lsflag < 0) {
      log.log_newton_failed(iter + 1, res_norm, newton_result_reason(NewtonResult::LSOLVE_FAIL));
      return NewtonResult::LSOLVE_FAIL;
    }

    delnorm = WrmsNorm(delta_data_);

    // Convergence: WRMS(delta, ewt) <= epsNewt
    if (delnorm <= epsNewt) {
      converged = true;
      if (delnorm <= abstolStep) {
        SaveIterate();
        RevertAndApply(SUN_RCONST(1.0), cj);
        log.log_newton_converged(iter + 1, newton_result_reason(NewtonResult::CONVERGED_WRMS_AND_STEPTOL));
        return NewtonResult::CONVERGED_WRMS_AND_STEPTOL;
      }
      if (iter > 0 && res_norm >= prev_res_norm) {
        RevertAndApply(SUN_RCONST(0.0), cj);
        log.log_newton_converged(iter + 1, newton_result_reason(NewtonResult::CONVERGED_WRMS_STEP_DIVERGED));
        return NewtonResult::CONVERGED_WRMS_STEP_DIVERGED;
      }
    }

    prev_res_norm = res_norm;
    SaveIterate();

    // Line search with Armijo condition
    sunrealtype alpha = SUN_RCONST(1.0);
    while (true) {
      RevertAndApply(alpha, cj);

      sunrealtype trial_norm = EvalResidualAndNorm(t);

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

  ComputeEwt();
  sunrealtype final_norm = EvalResidualAndNorm(t);
  log.log_newton_failed(max_iter, final_norm, newton_result_reason(NewtonResult::MAX_ITER_NO_CONVERGE));
  return NewtonResult::MAX_ITER_NO_CONVERGE;
}

// Top-level solve dispatch

template <class ExprSet>
NewtonResult NewtonSolver<ExprSet>::solve(
  sunrealtype t, sunrealtype t_next, SolverLog& log
) {
  if (len_alg_ <= 0) return NewtonResult::CONVERGED_WRMS_AND_STEPTOL;

  log.log_newton_start(t, len_alg_);
  SetupATimes();

  // Set active data pointers based on mode
  if (solve_type_ == NewtonSolveType::DECOUPLED_SUBBLOCK) {
    res_data_ = NV_DATA(res_alg_vec_);
    delta_data_ = NV_DATA(delta_alg_vec_);
  } else {
    res_data_ = NV_DATA(res_full_vec_);
    delta_data_ = NV_DATA(delta_full_vec_);
  }

  NewtonResult result;

  if (solve_type_ == NewtonSolveType::COUPLED_FULL) {
    sunrealtype hic = SUN_RCONST(0.001) * std::abs(t_next - t);
    if (hic == SUN_RCONST(0.0)) hic = SUN_RCONST(1.0e-6);
    const int max_nh = solver_opts_.max_num_steps_ic > 0;

    sunrealtype* y_val = NV_DATA(yy_);
    sunrealtype* yp_val = NV_DATA(yyp_);
    std::memcpy(y0_save_.data(), y_val, n_states_ * sizeof(sunrealtype));
    std::memcpy(yp0_save_.data(), yp_val, n_states_ * sizeof(sunrealtype));

    result = NewtonResult::MAX_ITER_NO_CONVERGE;
    for (int nh = 0; nh < max_nh; nh++) {
      if (nh > 0) {
        std::memcpy(y_val, y0_save_.data(), n_states_ * sizeof(sunrealtype));
        std::memcpy(yp_val, yp0_save_.data(), n_states_ * sizeof(sunrealtype));
      }
      sunrealtype cj = SUN_RCONST(1.0) / hic;
      result = RunNewtonLoop(t, cj, log);
      if (newton_success(result)) break;
      hic *= SUN_RCONST(0.1);
    }
  } else {
    sunrealtype cj = (solve_type_ == NewtonSolveType::DECOUPLED_FULL)
                       ? SUN_RCONST(1.0) : SUN_RCONST(0.0);
    result = RunNewtonLoop(t, cj, log);
  }

  RestoreATimes();
  return result;
}
