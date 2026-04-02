#pragma once

// Algebraic IC solver construction and helpers.
// Included from IDAKLUSolverOpenMP.hpp after the class definition.

#include "Expressions/Expressions.hpp"
#include "sundials_functions.hpp"

// ────────────────────── IDA linear-solve helper ──────────────────────

template <class ExprSet>
int IDAKLUSolverOpenMP<ExprSet>::SolveViaIDALinearSolver(
    N_Vector yy_ptr, N_Vector yyp_ptr,
    const sunrealtype* y_in, sunrealtype* res, sunrealtype* delta,
    sunrealtype cj) {

  IDAInternals ida(this->ida_mem);
  auto& fs = *this->alg_state_->full;

  sunrealtype* yy_data = NV_DATA(yy_ptr);
  std::memcpy(yy_data, y_in, number_of_states * sizeof(sunrealtype));

  ida.SetCj(cj);

  sunrealtype* res_data = N_VGetArrayPointer(fs.res_nvec);
  sunrealtype* delta_data = N_VGetArrayPointer(fs.delta_nvec);
  std::memcpy(res_data, res, number_of_states * sizeof(sunrealtype));

  int flag = ida.LinearSetup(yy_ptr, yyp_ptr, fs.res_nvec);
  if (flag != 0) return 1;

  std::memcpy(delta_data, res, number_of_states * sizeof(sunrealtype));
  flag = ida.LinearSolve(fs.delta_nvec, yy_ptr, yyp_ptr, fs.res_nvec);

  std::memcpy(delta, delta_data, number_of_states * sizeof(sunrealtype));
  return flag;
}

// ────────────────────── NonlinearSystem implementations ──────────────────────

template <class ExprSet>
class SubBlockSystem : public NonlinearSystem {
public:
  SubBlockSystem(
    ExprSet* funcs, N_Vector yy,
    IDAKLUSolverOpenMP<ExprSet>* solver,
    bool use_sparse)
    : funcs_(funcs), yy_(yy), solver_(solver), use_sparse_(use_sparse) {}

  void eval_residual(sunrealtype t, const sunrealtype* x_alg, sunrealtype* res_out) override {
    sunrealtype* yy_data = NV_DATA(yy_);
    std::memcpy(yy_data + solver_->len_rhs_, x_alg,
                solver_->len_alg_ * sizeof(sunrealtype));
    funcs_->alg_res->m_arg[0] = &t;
    funcs_->alg_res->m_arg[1] = yy_data;
    funcs_->alg_res->m_arg[2] = funcs_->inputs.data();
    funcs_->alg_res->m_res[0] = res_out;
    (*funcs_->alg_res)();
  }

  int solve_linear(sunrealtype t, const sunrealtype* x_alg,
                   sunrealtype* res, sunrealtype* delta) override {
    auto& sb = *solver_->alg_state_->sub;
    sunrealtype* yy_data = NV_DATA(yy_);
    std::memcpy(yy_data + solver_->len_rhs_, x_alg,
                solver_->len_alg_ * sizeof(sunrealtype));

    funcs_->alg_jac->m_arg[0] = &t;
    funcs_->alg_jac->m_arg[1] = yy_data;
    funcs_->alg_jac->m_arg[2] = funcs_->inputs.data();
    funcs_->alg_jac->m_res[0] = sb.full_jac_buf.data();
    (*funcs_->alg_jac)();

    if (use_sparse_) {
      sunrealtype* sub_data = SUNSparseMatrix_Data(sb.J);
      for (int i = 0; i < sb.nnz; i++)
        sub_data[i] = sb.full_jac_buf[sb.data_indices[i]];
    } else {
      sunrealtype* dense = SUNDenseMatrix_Data(sb.J);
      int len_alg = solver_->len_alg_;
      std::memset(dense, 0, len_alg * len_alg * sizeof(sunrealtype));
      for (int col = 0; col < len_alg; col++) {
        for (auto k = sb.colptrs[col]; k < sb.colptrs[col + 1]; k++) {
          int row = sb.rowvals[k];
          dense[col * len_alg + row] = sb.full_jac_buf[sb.data_indices[k]];
        }
      }
    }

    int flag = SUNLinSolSetup(sb.LS, sb.J);
    if (flag != 0) return 1;

    sunrealtype* res_data = N_VGetArrayPointer(sb.res_nvec);
    sunrealtype* delta_data = N_VGetArrayPointer(sb.delta_nvec);
    std::memcpy(res_data, res, solver_->len_alg_ * sizeof(sunrealtype));

    flag = SUNLinSolSolve(sb.LS, sb.J, sb.delta_nvec,
                          sb.res_nvec, SUN_RCONST(0.0));

    std::memcpy(delta, delta_data, solver_->len_alg_ * sizeof(sunrealtype));
    return flag;
  }

private:
  ExprSet* funcs_;
  N_Vector yy_;
  IDAKLUSolverOpenMP<ExprSet>* solver_;
  bool use_sparse_;
};

template <class ExprSet>
class DecoupledFullSystem : public NonlinearSystem {
public:
  DecoupledFullSystem(
    ExprSet* funcs, N_Vector yy, N_Vector yyp,
    IDAKLUSolverOpenMP<ExprSet>* solver)
    : funcs_(funcs), yy_(yy), yyp_(yyp), solver_(solver) {}

  void eval_residual(sunrealtype t, const sunrealtype* y, sunrealtype* res_out) override {
    sunrealtype* yy_data = NV_DATA(yy_);
    std::memcpy(yy_data, y, solver_->number_of_states * sizeof(sunrealtype));
    funcs_->rhs_alg->m_arg[0] = &t;
    funcs_->rhs_alg->m_arg[1] = yy_data;
    funcs_->rhs_alg->m_arg[2] = funcs_->inputs.data();
    funcs_->rhs_alg->m_res[0] = res_out;
    (*funcs_->rhs_alg)();
  }

  int solve_linear(sunrealtype t, const sunrealtype* y,
                   sunrealtype* res, sunrealtype* delta) override {
    return solver_->SolveViaIDALinearSolver(
      yy_, yyp_, y, res, delta, SUN_RCONST(1.0));
  }

private:
  ExprSet* funcs_;
  N_Vector yy_;
  N_Vector yyp_;
  IDAKLUSolverOpenMP<ExprSet>* solver_;
};

template <class ExprSet>
class CoupledFullSystem : public NonlinearSystem {
public:
  CoupledFullSystem(
    ExprSet* funcs, N_Vector yy, N_Vector yyp,
    IDAKLUSolverOpenMP<ExprSet>* solver)
    : funcs_(funcs), yy_(yy), yyp_(yyp), solver_(solver) {}

  void eval_residual(sunrealtype t, const sunrealtype* y, sunrealtype* res_out) override {
    int n = solver_->number_of_states;
    sunrealtype* yy_data = NV_DATA(yy_);
    sunrealtype* yp_data = NV_DATA(yyp_);
    std::memcpy(yy_data, y, n * sizeof(sunrealtype));
    update_yp(yy_data, yp_data);

    funcs_->rhs_alg->m_arg[0] = &t;
    funcs_->rhs_alg->m_arg[1] = yy_data;
    funcs_->rhs_alg->m_arg[2] = funcs_->inputs.data();
    funcs_->rhs_alg->m_res[0] = res_out;
    (*funcs_->rhs_alg)();

    sunrealtype* tmp = funcs_->get_tmp_state_vector();
    funcs_->mass_action->m_arg[0] = yp_data;
    funcs_->mass_action->m_res[0] = tmp;
    (*funcs_->mass_action)();
    axpy(n, -1., tmp, res_out);
  }

  int solve_linear(sunrealtype t, const sunrealtype* y,
                   sunrealtype* res, sunrealtype* delta) override {
    int n = solver_->number_of_states;
    sunrealtype* yy_data = NV_DATA(yy_);
    sunrealtype* yp_data = NV_DATA(yyp_);
    std::memcpy(yy_data, y, n * sizeof(sunrealtype));
    update_yp(yy_data, yp_data);

    auto& as = *solver_->alg_state_;
    return solver_->SolveViaIDALinearSolver(
      yy_, yyp_, y, res, delta, as.newton_cj);
  }

private:
  // yp[i] = yp0[i] + cj * (y[i] - y0[i]) for differential variables
  void update_yp(const sunrealtype* yy_data, sunrealtype* yp_data) {
    auto& as = *solver_->alg_state_;
    for (int idx : as.diff_idx) {
      yp_data[idx] = as.yp0_save_ic[idx]
        + as.newton_cj * (yy_data[idx] - as.y0_save_ic[idx]);
    }
  }

  ExprSet* funcs_;
  N_Vector yy_;
  N_Vector yyp_;
  IDAKLUSolverOpenMP<ExprSet>* solver_;
};

// ────────────────────── Mass-matrix alignment check ──────────────────────

template <class ExprSet>
bool IDAKLUSolverOpenMP<ExprSet>::CheckMassMatrixAlignment(const sunrealtype* id_val) {
  std::vector<sunrealtype> e_in(number_of_states, 0.0);
  std::vector<sunrealtype> m_out(number_of_states, 0.0);

  for (int j = 0; j < number_of_states; j++) {
    e_in[j] = 1.0;
    functions->mass_action->m_arg[0] = e_in.data();
    functions->mass_action->m_res[0] = m_out.data();
    (*functions->mass_action)();
    e_in[j] = 0.0;

    for (int i = 0; i < number_of_states; i++) {
      if (is_algebraic(id_val[i]) && m_out[i] != 0.0) {
        return false;
      }
    }
  }
  return true;
}

// ────────────────────── Sub-block sparsity pre-computation ──────────────────────

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::PrecomputeSubBlockSparsity() {
  auto& sb = *alg_state_->sub;
  Expression* jac_expr = functions->alg_jac;
  const auto& rows = jac_expr->get_row();
  const auto& cols = jac_expr->get_col();
  int nnz_total = jac_expr->nnz_out();

  // Build column pointers from COO data in O(nnz) instead of O(n_alg * nnz).
  // The full Jacobian has number_of_states columns; we only need the algebraic
  // columns [len_rhs_, len_rhs_ + len_alg_).
  int n_full_cols = number_of_states;
  std::vector<int> col_count(n_full_cols + 1, 0);
  for (int k = 0; k < nnz_total; k++)
    col_count[static_cast<int>(cols[k]) + 1]++;

  // Prefix sum -> column pointers
  for (int c = 1; c <= n_full_cols; c++)
    col_count[c] += col_count[c - 1];

  // Scatter into column-sorted order
  std::vector<int> sorted_row(nnz_total);
  std::vector<int> sorted_orig_idx(nnz_total);
  std::vector<int> pos(col_count);  // copy for scatter
  for (int k = 0; k < nnz_total; k++) {
    int c = static_cast<int>(cols[k]);
    int dest = pos[c]++;
    sorted_row[dest] = static_cast<int>(rows[k]);
    sorted_orig_idx[dest] = k;
  }

  // Extract only the algebraic columns
  sb.colptrs.resize(len_alg_ + 1);
  sb.rowvals.clear();
  sb.data_indices.clear();

  for (int alg_col = 0; alg_col < len_alg_; alg_col++) {
    sb.colptrs[alg_col] = static_cast<sunindextype>(sb.rowvals.size());
    int full_col = alg_col + len_rhs_;
    int start = col_count[full_col];
    int end = col_count[full_col + 1];
    for (int p = start; p < end; p++) {
      sb.rowvals.push_back(static_cast<sunindextype>(sorted_row[p]));
      sb.data_indices.push_back(sorted_orig_idx[p]);
    }
  }
  sb.colptrs[len_alg_] = static_cast<sunindextype>(sb.rowvals.size());
  sb.nnz = static_cast<int>(sb.rowvals.size());
}

// ────────────────────── BuildAlgebraicSolver ──────────────────────

template <class ExprSet>
void IDAKLUSolverOpenMP<ExprSet>::BuildAlgebraicSolver(const sunrealtype* id_val) {
  DEBUG("IDAKLUSolverOpenMP::BuildAlgebraicSolver");

  alg_state_ = std::make_unique<AlgSolverState>();
  auto& as = *alg_state_;

  for (int i = 0; i < number_of_states; i++) {
    if (is_algebraic(id_val[i]))
      as.alg_idx.push_back(i);
    else
      as.diff_idx.push_back(i);
  }

  bool mass_aligned = CheckMassMatrixAlignment(id_val);
  as.is_coupled = !mass_aligned;

  auto* funcs = functions.get();
  auto* yy_ptr = yy;
  auto* yyp_ptr = yyp;
  const sunrealtype* atol_data = N_VGetArrayPointer(avtol);

  using Mode = AlgSolverState::Mode;
  bool has_alg_fns = (funcs->alg_res->nnz_out() > 0 && funcs->alg_jac->nnz_out() > 0);
  if (as.is_coupled) {
    as.mode = Mode::COUPLED_FULL;
  } else if (has_alg_fns) {
    as.mode = Mode::SUBBLOCK;
  } else {
    as.mode = Mode::DECOUPLED_FULL;
  }

  int n_solve_vars;
  std::vector<int> solve_diff_idx;

  if (as.mode == Mode::SUBBLOCK) {
    // ── SUBBLOCK: solve only algebraic variables with own LS/J ──
    if (funcs->alg_res->nnz_out() == 0 || funcs->alg_jac->nnz_out() == 0) {
      throw std::runtime_error(
        "BuildAlgebraicSolver: SUBBLOCK mode requires alg_res and alg_jac functions");
    }

    n_solve_vars = len_alg_;

    bool use_sparse = (setup_opts.jacobian == "sparse" &&
                       setup_opts.linear_solver == "SUNLinSol_KLU");

    as.sub = std::make_unique<SubBlockResources>();
    auto& sb = *as.sub;

    SUNContext_Create(SUN_COMM_NULL, &sb.sunctx);
    PrecomputeSubBlockSparsity();

    int alg_jac_nnz = funcs->alg_jac->nnz_out();
    sb.full_jac_buf.resize(alg_jac_nnz > 0
      ? alg_jac_nnz : len_alg_ * number_of_states);

    sb.res_nvec = N_VNew_Serial(len_alg_, sb.sunctx);
    sb.delta_nvec = N_VNew_Serial(len_alg_, sb.sunctx);

    if (use_sparse) {
      sb.J = SUNSparseMatrix(len_alg_, len_alg_, sb.nnz, CSC_MAT, sb.sunctx);
      sunindextype* jp = SUNSparseMatrix_IndexPointers(sb.J);
      sunindextype* jr = SUNSparseMatrix_IndexValues(sb.J);
      for (int i = 0; i <= len_alg_; i++) jp[i] = sb.colptrs[i];
      for (int i = 0; i < sb.nnz; i++) jr[i] = sb.rowvals[i];
      sb.LS = SUNLinSol_KLU(sb.delta_nvec, sb.J, sb.sunctx);
    } else {
      sb.J = SUNDenseMatrix(len_alg_, len_alg_, sb.sunctx);
      sb.LS = SUNLinSol_Dense(sb.delta_nvec, sb.J, sb.sunctx);
    }

    as.system = std::make_unique<SubBlockSystem<ExprSet>>(
      funcs, yy_ptr, this, use_sparse);

  } else if (as.mode == Mode::DECOUPLED_FULL) {
    // ── DECOUPLED_FULL: full-system via IDA's linear solve interface ──
    n_solve_vars = number_of_states;
    solve_diff_idx = as.diff_idx;

    as.full = std::make_unique<FullSystemResources>();
    auto& fs = *as.full;
    fs.res_nvec = N_VNew_Serial(number_of_states, sunctx);
    fs.delta_nvec = N_VNew_Serial(number_of_states, sunctx);

    as.system = std::make_unique<DecoupledFullSystem<ExprSet>>(
      funcs, yy_ptr, yyp_ptr, this);

  } else {
    // ── COUPLED_FULL: solve full system, all variables participate ──
    n_solve_vars = number_of_states;
    solve_diff_idx = as.diff_idx;

    as.full = std::make_unique<FullSystemResources>();
    auto& fs = *as.full;
    fs.res_nvec = N_VNew_Serial(number_of_states, sunctx);
    fs.delta_nvec = N_VNew_Serial(number_of_states, sunctx);

    as.system = std::make_unique<CoupledFullSystem<ExprSet>>(
      funcs, yy_ptr, yyp_ptr, this);
  }

  // Build atol for the solve dimension
  std::vector<sunrealtype> solve_atol(n_solve_vars);
  if (as.mode == Mode::SUBBLOCK) {
    for (int i = 0; i < len_alg_; i++)
      solve_atol[i] = atol_data[as.alg_idx[i]];
  } else {
    std::memcpy(solve_atol.data(), atol_data, number_of_states * sizeof(sunrealtype));
  }

  as.solver = std::make_unique<NonlinearSolver>(
    *as.system,
    n_solve_vars,
    solve_atol.data(),
    rtol,
    solver_opts.newton_step_tol,
    solver_opts.max_num_iterations_ic,
    solver_opts.max_linesearch_backtracks_ic,
    solver_opts.nonlinear_convergence_coefficient_ic,
    solve_diff_idx,
    as.is_coupled
  );
  as.solver->set_log(&log_);

  as.y_backup.resize(number_of_states);
  as.yp_backup.resize(number_of_states);

  if (as.is_coupled) {
    as.y0_save_ic.resize(number_of_states);
    as.yp0_save_ic.resize(number_of_states);
  }
}
