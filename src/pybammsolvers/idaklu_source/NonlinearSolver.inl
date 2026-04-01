#pragma once

// ────────────────────── Standalone constructor ──────────────────────

inline NonlinearSolver::NonlinearSolver(
  const casadi::Function& residual_fn,
  const casadi::Function& jacobian_fn,
  int n_vars,
  np_array atol_np,
  double rtol,
  double step_tol,
  int inputs_length,
  py::dict options
) : n_vars_(n_vars),
    rtol_(rtol),
    step_tol_(step_tol),
    ls_(nullptr),
    use_sparse_(false),
    jac_nnz_(0),
    last_num_iterations_(0)
{
  max_iter_ = options.contains("max_iterations")
    ? options["max_iterations"].cast<int>() : 100;
  max_backtracks_ = options.contains("max_linesearch_backtracks")
    ? options["max_linesearch_backtracks"].cast<int>() : 100;
  epsNewt_ = options.contains("epsNewt")
    ? options["epsNewt"].cast<sunrealtype>() : SUN_RCONST(0.0033);

  std::string jacobian_type = options.contains("jacobian")
    ? options["jacobian"].cast<std::string>() : "sparse";
  std::string linear_solver = options.contains("linear_solver")
    ? options["linear_solver"].cast<std::string>() : "SUNLinSol_KLU";

  use_sparse_ = (jacobian_type == "sparse" && linear_solver == "SUNLinSol_KLU");

  // Copy atol
  auto atol_buf = atol_np.request();
  atol_.resize(n_vars_);
  if (atol_buf.size == 1) {
    sunrealtype val = static_cast<sunrealtype*>(atol_buf.ptr)[0];
    std::fill(atol_.begin(), atol_.end(), val);
  } else {
    std::memcpy(atol_.data(), atol_buf.ptr, n_vars_ * sizeof(sunrealtype));
  }

  // Pre-allocate all working vectors
  x_.resize(n_vars_);
  res_.resize(n_vars_);
  delta_.resize(n_vars_);
  x_save_.resize(n_vars_);
  ewt_.resize(n_vars_);
  inputs_.resize(inputs_length);

  // Create CasADi function wrappers
  residual_casadi_ = std::make_unique<CasadiFunction>(residual_fn);
  jacobian_casadi_ = std::make_unique<CasadiFunction>(jacobian_fn);

  jac_nnz_ = jacobian_casadi_->nnz_out();
  jac_buf_.resize(jac_nnz_ > 0 ? jac_nnz_ : n_vars_ * n_vars_);

  // Create owned LinearSolver from Jacobian sparsity
  if (use_sparse_) {
    const auto& rows = jacobian_casadi_->get_row();
    const auto& cols = jacobian_casadi_->get_col();

    // Convert COO to CSC
    std::vector<int64_t> colptrs(n_vars_ + 1, 0);
    std::vector<int64_t> rowvals(jac_nnz_);

    std::vector<int> col_counts(n_vars_, 0);
    for (int k = 0; k < jac_nnz_; k++) {
      col_counts[cols[k]]++;
    }
    colptrs[0] = 0;
    for (int c = 0; c < n_vars_; c++) {
      colptrs[c + 1] = colptrs[c] + col_counts[c];
    }
    std::vector<int> col_offsets(n_vars_, 0);
    for (int k = 0; k < jac_nnz_; k++) {
      int c = cols[k];
      int pos = colptrs[c] + col_offsets[c];
      rowvals[pos] = rows[k];
      col_offsets[c]++;
    }

    owned_ls_ = std::make_unique<LinearSolver>(
      n_vars_, jac_nnz_,
      colptrs.data(), rowvals.data(),
      options);
  } else {
    // Dense mode
    py::dict ls_opts;
    ls_opts["jacobian"] = "dense";
    ls_opts["linear_solver"] = "SUNLinSol_Dense";
    owned_ls_ = std::make_unique<LinearSolver>(
      n_vars_, 0, nullptr, nullptr, ls_opts);
  }
  ls_ = owned_ls_.get();

  // Set up callbacks that use CasADi functions
  eval_residual_ = [this](sunrealtype t, const sunrealtype* x, sunrealtype* res_out) {
    residual_casadi_->m_arg[0] = &t;
    residual_casadi_->m_arg[1] = const_cast<sunrealtype*>(x);
    residual_casadi_->m_arg[2] = inputs_.data();
    residual_casadi_->m_res[0] = res_out;
    (*residual_casadi_)();
  };

  eval_jacobian_ = [this](sunrealtype t, const sunrealtype* x) {
    jacobian_casadi_->m_arg[0] = &t;
    jacobian_casadi_->m_arg[1] = const_cast<sunrealtype*>(x);
    jacobian_casadi_->m_arg[2] = inputs_.data();
    jacobian_casadi_->m_res[0] = jac_buf_.data();
    (*jacobian_casadi_)();

    if (use_sparse_) {
      ls_->factorize(jac_buf_.data());
    } else {
      // Scatter sparse CasADi output into dense matrix if needed
      if (jac_nnz_ > 0 && jac_nnz_ < n_vars_ * n_vars_) {
        std::vector<sunrealtype> dense(n_vars_ * n_vars_, SUN_RCONST(0.0));
        const auto& rows = jacobian_casadi_->get_row();
        const auto& cols = jacobian_casadi_->get_col();
        for (int k = 0; k < jac_nnz_; k++) {
          dense[cols[k] * n_vars_ + rows[k]] = jac_buf_[k];
        }
        ls_->factorize(dense.data());
      } else {
        ls_->factorize(jac_buf_.data());
      }
    }
  };
}

// ────────────────────── Borrowed constructor ──────────────────────

inline NonlinearSolver::NonlinearSolver(
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
) : n_vars_(n_vars),
    rtol_(rtol),
    step_tol_(step_tol),
    max_iter_(max_iter),
    max_backtracks_(max_backtracks),
    epsNewt_(epsNewt),
    ls_(ls),
    eval_residual_(std::move(eval_residual)),
    eval_jacobian_(std::move(eval_jacobian)),
    use_sparse_(false),
    jac_nnz_(0),
    last_num_iterations_(0)
{
  atol_.resize(n_vars_);
  std::memcpy(atol_.data(), atol_data, n_vars_ * sizeof(sunrealtype));

  x_.resize(n_vars_);
  res_.resize(n_vars_);
  delta_.resize(n_vars_);
  x_save_.resize(n_vars_);
  ewt_.resize(n_vars_);
}

// ────────────────────── Destructor ──────────────────────

inline NonlinearSolver::~NonlinearSolver() = default;

// ────────────────────── Helpers ──────────────────────

inline sunrealtype NonlinearSolver::WrmsNorm(const sunrealtype* vals) const {
  sunrealtype sum = SUN_RCONST(0.0);
  for (int i = 0; i < n_vars_; i++) {
    sunrealtype w = vals[i] * ewt_[i];
    sum += w * w;
  }
  return std::sqrt(sum / n_vars_);
}

inline sunrealtype NonlinearSolver::InfNorm(const sunrealtype* vals) const {
  sunrealtype mx = SUN_RCONST(0.0);
  for (int i = 0; i < n_vars_; i++) {
    sunrealtype a = std::abs(vals[i]);
    if (a > mx) mx = a;
  }
  return mx;
}

inline void NonlinearSolver::ComputeEwt() {
  for (int i = 0; i < n_vars_; i++) {
    ewt_[i] = SUN_RCONST(1.0) / (rtol_ * std::abs(x_[i]) + atol_[i]);
  }
}

inline void NonlinearSolver::SaveIterate() {
  std::memcpy(x_save_.data(), x_.data(), n_vars_ * sizeof(sunrealtype));
}

inline void NonlinearSolver::RevertAndApply(sunrealtype alpha) {
  for (int i = 0; i < n_vars_; i++) {
    x_[i] = x_save_[i] - alpha * delta_[i];
  }
}

// ────────────────────── Evaluate residual ──────────────────────

inline sunrealtype NonlinearSolver::EvalResidualAndNorm(sunrealtype t) {
  eval_residual_(t, x_.data(), res_.data());
  return InfNorm(res_.data());
}

// ────────────────────── Evaluate Jacobian + linear solve ──────────────────────

inline int NonlinearSolver::SetupAndSolveLinearSystem(sunrealtype t) {
  // Evaluate Jacobian and factorize (callback handles both)
  try {
    eval_jacobian_(t, x_.data());
  } catch (...) {
    return 1;  // LSETUP_FAIL
  }

  // Solve J*delta = res
  try {
    ls_->solve(res_.data(), delta_.data());
  } catch (...) {
    return -1;  // LSOLVE_FAIL
  }

  return 0;
}

// ────────────────────── THE Newton loop (written ONCE) ──────────────────────

inline NonlinearResult NonlinearSolver::RunNewtonLoop(sunrealtype t) {
  sunrealtype delnorm = std::numeric_limits<sunrealtype>::infinity();
  sunrealtype prev_res_norm = std::numeric_limits<sunrealtype>::infinity();
  bool converged = false;

  ComputeEwt();

  for (int iter = 0; iter < max_iter_; iter++) {
    sunrealtype res_norm = EvalResidualAndNorm(t);

    int lsflag = SetupAndSolveLinearSystem(t);
    if (lsflag > 0) {
      last_message_ = nonlinear_result_reason(NonlinearResult::LSETUP_FAIL);
      last_num_iterations_ = iter + 1;
      return NonlinearResult::LSETUP_FAIL;
    }
    if (lsflag < 0) {
      last_message_ = nonlinear_result_reason(NonlinearResult::LSOLVE_FAIL);
      last_num_iterations_ = iter + 1;
      return NonlinearResult::LSOLVE_FAIL;
    }

    delnorm = WrmsNorm(delta_.data());

    if (delnorm <= epsNewt_) {
      converged = true;
      if (delnorm <= step_tol_) {
        SaveIterate();
        RevertAndApply(SUN_RCONST(1.0));
        last_message_ = nonlinear_result_reason(NonlinearResult::CONVERGED_WRMS_AND_STEPTOL);
        last_num_iterations_ = iter + 1;
        return NonlinearResult::CONVERGED_WRMS_AND_STEPTOL;
      }
      if (iter > 0 && res_norm >= prev_res_norm) {
        RevertAndApply(SUN_RCONST(0.0));
        last_message_ = nonlinear_result_reason(NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED);
        last_num_iterations_ = iter + 1;
        return NonlinearResult::CONVERGED_WRMS_STEP_DIVERGED;
      }
    }

    prev_res_norm = res_norm;
    SaveIterate();

    // Line search with Armijo condition
    sunrealtype alpha = SUN_RCONST(1.0);
    for (int ls = 0; ls < max_backtracks_; ls++) {
      RevertAndApply(alpha);
      sunrealtype trial_norm = EvalResidualAndNorm(t);
      if (trial_norm <= (SUN_RCONST(1.0) - alpha * SUN_RCONST(0.5)) * res_norm)
        break;
      if (alpha * delnorm <= step_tol_)
        break;
      alpha *= SUN_RCONST(0.5);
    }
  }

  if (converged) {
    last_message_ = nonlinear_result_reason(NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER);
    last_num_iterations_ = max_iter_;
    return NonlinearResult::CONVERGED_WRMS_AT_MAX_ITER;
  }

  last_message_ = nonlinear_result_reason(NonlinearResult::MAX_ITER_NO_CONVERGE);
  last_num_iterations_ = max_iter_;
  return NonlinearResult::MAX_ITER_NO_CONVERGE;
}

// ────────────────────── solve_single (C++ in-place) ──────────────────────

inline NonlinearResult NonlinearSolver::solve_single(
  sunrealtype t, sunrealtype* x, const sunrealtype* p
) {
  // Copy initial guess into working buffer
  std::memcpy(x_.data(), x, n_vars_ * sizeof(sunrealtype));

  // Copy parameters if in standalone mode
  if (p != nullptr && !inputs_.empty()) {
    std::memcpy(inputs_.data(), p, inputs_.size() * sizeof(sunrealtype));
  }

  NonlinearResult result = RunNewtonLoop(t);

  // Copy solution back (in-place)
  std::memcpy(x, x_.data(), n_vars_ * sizeof(sunrealtype));

  return result;
}

// ────────────────────── solve (Python-facing batch) ──────────────────────

inline py::object NonlinearSolver::solve(
  np_array t_eval_np,
  np_array x0_np,
  np_array inputs_np,
  py::object out
) {
  auto t_buf = t_eval_np.request();
  auto x0_buf = x0_np.request();
  auto inputs_buf = inputs_np.request();

  const int n_times = static_cast<int>(t_buf.size);
  const sunrealtype* t_eval = static_cast<sunrealtype*>(t_buf.ptr);

  // Copy initial guess
  std::memcpy(x_.data(), x0_buf.ptr, n_vars_ * sizeof(sunrealtype));

  // Copy inputs
  if (inputs_buf.size > 0) {
    std::memcpy(inputs_.data(), inputs_buf.ptr,
                inputs_buf.size * sizeof(sunrealtype));
  }

  // Determine output storage
  sunrealtype* x_out_ptr;
  np_array x_result;
  bool using_out = !out.is_none();

  if (using_out) {
    x_result = out.cast<np_array>();
    auto out_buf = x_result.request();
    if (out_buf.ndim != 2 || out_buf.shape[0] != n_vars_ || out_buf.shape[1] != n_times) {
      throw std::runtime_error(
        "NonlinearSolver::solve: out must have shape (n_vars, n_times)");
    }
    x_out_ptr = static_cast<sunrealtype*>(out_buf.ptr);
  } else {
    x_result = np_array({n_vars_, n_times});
    x_out_ptr = x_result.mutable_data();
  }

  // Allocate t output
  auto t_np = py::array_t<sunrealtype>(n_times);
  sunrealtype* t_out = t_np.mutable_data();

  bool all_success = true;
  int failed_idx = -1;
  int n_solved = 0;

  for (int i = 0; i < n_times; i++) {
    NonlinearResult result = RunNewtonLoop(t_eval[i]);

    t_out[i] = t_eval[i];
    // Copy x_ into column i of x_out (column-major: x_out[j * n_times + i])
    // But np_array is row-major by default. We use (n_vars, n_times) layout.
    for (int j = 0; j < n_vars_; j++) {
      x_out_ptr[j * n_times + i] = x_[j];
    }
    n_solved = i + 1;

    if (!nonlinear_success(result)) {
      all_success = false;
      failed_idx = i;
      break;
    }
  }

  // Trim if we stopped early
  if (n_solved < n_times) {
    t_np = py::array_t<sunrealtype>(n_solved);
    std::memcpy(t_np.mutable_data(), t_out, n_solved * sizeof(sunrealtype));

    x_result = np_array({n_vars_, n_solved});
    sunrealtype* x_dst = x_result.mutable_data();
    for (int j = 0; j < n_vars_; j++) {
      for (int i = 0; i < n_solved; i++) {
        x_dst[j * n_solved + i] = x_out_ptr[j * n_times + i];
      }
    }
  }

  int flag = all_success ? 0 : -1;
  std::string message = all_success
    ? last_message_
    : ("Newton solver failed at t=" + std::to_string(t_eval[failed_idx]) +
       ": " + last_message_);

  py::dict result_dict;
  result_dict["t"] = t_np;
  result_dict["x"] = x_result;
  result_dict["flag"] = flag;
  result_dict["success"] = all_success;
  result_dict["message"] = message;
  result_dict["num_iterations"] = last_num_iterations_;

  return result_dict;
}
