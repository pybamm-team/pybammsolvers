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
    sunrealtype cj, bool update_yp) {

  IDAInternals ida(this->ida_mem);
  auto& fs = *this->alg_state_->full;

  if (!update_yp) {
    sunrealtype* yy_data = NV_DATA(yy_ptr);
    std::memcpy(yy_data, y_in, number_of_states * sizeof(sunrealtype));
  }

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

  sb.colptrs.resize(len_alg_ + 1, 0);
  sb.rowvals.clear();
  sb.data_indices.clear();

  for (int alg_col = 0; alg_col < len_alg_; alg_col++) {
    sb.colptrs[alg_col] = static_cast<sunindextype>(sb.rowvals.size());
    int full_col = alg_col + len_rhs_;
    for (int k = 0; k < nnz_total; k++) {
      if (static_cast<int>(cols[k]) == full_col) {
        sb.rowvals.push_back(static_cast<sunindextype>(rows[k]));
        sb.data_indices.push_back(k);
      }
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
  if (!as.is_coupled && has_alg_fns) {
    as.mode = Mode::SUBBLOCK;
  } else if (!as.is_coupled) {
    as.mode = Mode::DECOUPLED_FULL;
  } else {
    as.mode = Mode::COUPLED_FULL;
  }

  NonlinearSolver::ResidualFn res_fn;
  NonlinearSolver::LinearSolveFn solve_fn;
  int n_solve_vars;
  std::vector<int> solve_diff_idx;

  if (as.mode == AlgSolverState::Mode::SUBBLOCK) {
    // ── SUBBLOCK: solve only algebraic variables with own LS/J ──
    if (funcs->alg_res->nnz_out() == 0 || funcs->alg_jac->nnz_out() == 0) {
      throw std::runtime_error(
        "BuildAlgebraicSolver: SUBBLOCK mode requires alg_res and alg_jac functions");
    }

    n_solve_vars = len_alg_;

    res_fn = [funcs, yy_ptr, this](sunrealtype t, const sunrealtype* x_alg, sunrealtype* res_out) {
      sunrealtype* yy_data = NV_DATA(yy_ptr);
      std::memcpy(yy_data + len_rhs_, x_alg, len_alg_ * sizeof(sunrealtype));
      funcs->alg_res->m_arg[0] = &t;
      funcs->alg_res->m_arg[1] = yy_data;
      funcs->alg_res->m_arg[2] = funcs->inputs.data();
      funcs->alg_res->m_res[0] = res_out;
      (*funcs->alg_res)();
    };

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

    solve_fn = [funcs, yy_ptr, use_sparse, this](
      sunrealtype t, const sunrealtype* x_alg,
      sunrealtype* res, sunrealtype* delta) -> int {

      auto& sb = *this->alg_state_->sub;
      sunrealtype* yy_data = NV_DATA(yy_ptr);
      std::memcpy(yy_data + len_rhs_, x_alg, len_alg_ * sizeof(sunrealtype));

      funcs->alg_jac->m_arg[0] = &t;
      funcs->alg_jac->m_arg[1] = yy_data;
      funcs->alg_jac->m_arg[2] = funcs->inputs.data();
      funcs->alg_jac->m_res[0] = sb.full_jac_buf.data();
      (*funcs->alg_jac)();

      if (use_sparse) {
        sunrealtype* sub_data = SUNSparseMatrix_Data(sb.J);
        for (int i = 0; i < sb.nnz; i++)
          sub_data[i] = sb.full_jac_buf[sb.data_indices[i]];
      } else {
        sunrealtype* dense = SUNDenseMatrix_Data(sb.J);
        std::memset(dense, 0, len_alg_ * len_alg_ * sizeof(sunrealtype));
        for (int col = 0; col < len_alg_; col++) {
          for (auto k = sb.colptrs[col]; k < sb.colptrs[col + 1]; k++) {
            int row = sb.rowvals[k];
            dense[col * len_alg_ + row] =
              sb.full_jac_buf[sb.data_indices[k]];
          }
        }
      }

      int flag = SUNLinSolSetup(sb.LS, sb.J);
      if (flag != 0) return 1;

      sunrealtype* res_data = N_VGetArrayPointer(sb.res_nvec);
      sunrealtype* delta_data = N_VGetArrayPointer(sb.delta_nvec);
      std::memcpy(res_data, res, len_alg_ * sizeof(sunrealtype));

      flag = SUNLinSolSolve(sb.LS, sb.J, sb.delta_nvec,
                            sb.res_nvec, SUN_RCONST(0.0));

      std::memcpy(delta, delta_data, len_alg_ * sizeof(sunrealtype));
      return flag;
    };

  } else if (as.mode == AlgSolverState::Mode::DECOUPLED_FULL) {
    // ── DECOUPLED_FULL: full-system via IDA's linear solve interface ──
    n_solve_vars = number_of_states;
    solve_diff_idx = as.diff_idx;

    as.full = std::make_unique<FullSystemResources>();
    auto& fs = *as.full;

    res_fn = [funcs, yy_ptr, this](sunrealtype t, const sunrealtype* y, sunrealtype* res_out) {
      sunrealtype* yy_data = NV_DATA(yy_ptr);
      std::memcpy(yy_data, y, number_of_states * sizeof(sunrealtype));
      funcs->rhs_alg->m_arg[0] = &t;
      funcs->rhs_alg->m_arg[1] = yy_data;
      funcs->rhs_alg->m_arg[2] = funcs->inputs.data();
      funcs->rhs_alg->m_res[0] = res_out;
      (*funcs->rhs_alg)();
    };

    fs.res_nvec = N_VNew_Serial(number_of_states, sunctx);
    fs.delta_nvec = N_VNew_Serial(number_of_states, sunctx);

    solve_fn = [yy_ptr, yyp_ptr, this](sunrealtype t, const sunrealtype* y,
                                        sunrealtype* res, sunrealtype* delta) -> int {
      return SolveViaIDALinearSolver(
        yy_ptr, yyp_ptr, y, res, delta, SUN_RCONST(1.0), /*update_yp=*/false);
    };

  } else {
    // ── COUPLED_FULL: solve full system, all variables participate ──
    n_solve_vars = number_of_states;
    solve_diff_idx = as.diff_idx;

    as.full = std::make_unique<FullSystemResources>();
    auto& fs = *as.full;

    res_fn = [funcs, yy_ptr, yyp_ptr, this](sunrealtype t, const sunrealtype* y, sunrealtype* res_out) {
      sunrealtype* yy_data = NV_DATA(yy_ptr);
      sunrealtype* yp_data = NV_DATA(yyp_ptr);
      std::memcpy(yy_data, y, number_of_states * sizeof(sunrealtype));
      for (int idx : this->alg_state_->diff_idx) {
        yp_data[idx] = this->alg_state_->yp0_save_ic[idx]
          + this->alg_state_->newton_cj * (yy_data[idx] - this->alg_state_->y0_save_ic[idx]);
      }
      funcs->rhs_alg->m_arg[0] = &t;
      funcs->rhs_alg->m_arg[1] = yy_data;
      funcs->rhs_alg->m_arg[2] = funcs->inputs.data();
      funcs->rhs_alg->m_res[0] = res_out;
      (*funcs->rhs_alg)();
      sunrealtype* tmp = funcs->get_tmp_state_vector();
      funcs->mass_action->m_arg[0] = yp_data;
      funcs->mass_action->m_res[0] = tmp;
      (*funcs->mass_action)();
      axpy(number_of_states, -1., tmp, res_out);
    };

    fs.res_nvec = N_VNew_Serial(number_of_states, sunctx);
    fs.delta_nvec = N_VNew_Serial(number_of_states, sunctx);

    solve_fn = [yy_ptr, yyp_ptr, this](sunrealtype t, const sunrealtype* y,
                                        sunrealtype* res, sunrealtype* delta) -> int {
      auto& as = *this->alg_state_;
      sunrealtype* yp_data = NV_DATA(yyp_ptr);
      sunrealtype* yy_data = NV_DATA(yy_ptr);
      std::memcpy(yy_data, y, number_of_states * sizeof(sunrealtype));
      for (int idx : as.diff_idx) {
        yp_data[idx] = as.yp0_save_ic[idx]
          + as.newton_cj * (yy_data[idx] - as.y0_save_ic[idx]);
      }
      return SolveViaIDALinearSolver(
        yy_ptr, yyp_ptr, y, res, delta, as.newton_cj, /*update_yp=*/true);
    };
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
    std::move(res_fn),
    std::move(solve_fn),
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

  if (as.is_coupled) {
    as.y0_save_ic.resize(number_of_states);
    as.yp0_save_ic.resize(number_of_states);
  }
}
