#ifndef PYBAMM_STANDALONE_NEWTON_SOLVER_HPP
#define PYBAMM_STANDALONE_NEWTON_SOLVER_HPP

#include "NonlinearSolver.hpp"
#include "Expressions/Casadi/CasadiFunctions.hpp"
#include <memory>
#include <utility>

/**
 * @brief NonlinearSystem backed by standalone CasadiFunctions + SUNLinSol.
 *
 * Self-contained: no dependency on IDAKLUSolverOpenMP or IDA memory.
 * Residual signature: F(t, y_alg, inputs) -> res   (n_vars outputs)
 * Jacobian signature: J(t, y_alg, inputs) -> data   (COO or dense)
 */
class StandaloneAlgebraicSystem : public NonlinearSystem {
public:
  StandaloneAlgebraicSystem(
    casadi::Function res_fn,
    casadi::Function jac_fn,
    bool use_sparse)
    : res_cf_(res_fn),
      jac_cf_(jac_fn),
      n_vars_(static_cast<int>(res_fn.nnz_out(0))),
      use_sparse_(use_sparse),
      sunctx_(nullptr), J_(nullptr), LS_(nullptr),
      res_nvec_(nullptr), delta_nvec_(nullptr)
  {
    SUNContext_Create(SUN_COMM_NULL, &sunctx_);

    res_nvec_ = N_VNew_Serial(n_vars_, sunctx_);
    delta_nvec_ = N_VNew_Serial(n_vars_, sunctx_);

    int jac_nnz = static_cast<int>(jac_cf_.nnz_out());
    jac_buf_.resize(jac_nnz > 0 ? jac_nnz : n_vars_ * n_vars_);

    if (use_sparse_ && jac_nnz > 0) {
      BuildSparseResources(jac_nnz);
    } else {
      use_sparse_ = false;
      BuildDenseResources();
    }
  }

  ~StandaloneAlgebraicSystem() {
    if (res_nvec_) N_VDestroy(res_nvec_);
    if (delta_nvec_) N_VDestroy(delta_nvec_);
    if (LS_) SUNLinSolFree(LS_);
    if (J_) SUNMatDestroy(J_);
    if (sunctx_) SUNContext_Free(&sunctx_);
  }

  int n_vars() const { return n_vars_; }

  void set_inputs(const sunrealtype* inputs_data, int inputs_len) {
    inputs_.assign(inputs_data, inputs_data + inputs_len);
  }

  void eval_residual(sunrealtype t, const sunrealtype* y,
                     sunrealtype* res) override {
    res_cf_.m_arg[0] = &t;
    res_cf_.m_arg[1] = y;
    res_cf_.m_arg[2] = inputs_.data();
    res_cf_.m_res[0] = res;
    res_cf_();
  }

  int solve_linear(sunrealtype t, const sunrealtype* y,
                   sunrealtype* res, sunrealtype* delta) override {
    jac_cf_.m_arg[0] = &t;
    jac_cf_.m_arg[1] = y;
    jac_cf_.m_arg[2] = inputs_.data();
    jac_cf_.m_res[0] = jac_buf_.data();
    jac_cf_();

    if (use_sparse_) {
      sunrealtype* mat_data = SUNSparseMatrix_Data(J_);
      for (int i = 0; i < nnz_; i++)
        mat_data[i] = jac_buf_[data_indices_[i]];
    } else {
      sunrealtype* dense = SUNDenseMatrix_Data(J_);
      std::memset(dense, 0, n_vars_ * n_vars_ * sizeof(sunrealtype));
      for (int col = 0; col < n_vars_; col++) {
        for (auto k = colptrs_[col]; k < colptrs_[col + 1]; k++) {
          int row = rowvals_[k];
          dense[col * n_vars_ + row] = jac_buf_[data_indices_[k]];
        }
      }
    }

    int flag = SUNLinSolSetup(LS_, J_);
    if (flag != 0) return 1;

    sunrealtype* res_data = N_VGetArrayPointer(res_nvec_);
    sunrealtype* delta_data = N_VGetArrayPointer(delta_nvec_);
    std::memcpy(res_data, res, n_vars_ * sizeof(sunrealtype));

    flag = SUNLinSolSolve(LS_, J_, delta_nvec_, res_nvec_, SUN_RCONST(0.0));

    std::memcpy(delta, delta_data, n_vars_ * sizeof(sunrealtype));
    return flag;
  }

private:
  void BuildSparseResources(int jac_nnz) {
    const auto& rows = jac_cf_.get_row();
    const auto& cols = jac_cf_.get_col();
    int nnz_total = jac_nnz;

    // Build CSC from COO (same O(nnz) algorithm as AlgebraicICBuilder)
    std::vector<int> col_count(n_vars_ + 1, 0);
    for (int k = 0; k < nnz_total; k++)
      col_count[static_cast<int>(cols[k]) + 1]++;

    for (int c = 1; c <= n_vars_; c++)
      col_count[c] += col_count[c - 1];

    std::vector<int> sorted_row(nnz_total);
    std::vector<int> sorted_orig_idx(nnz_total);
    std::vector<int> pos(col_count);
    for (int k = 0; k < nnz_total; k++) {
      int c = static_cast<int>(cols[k]);
      int dest = pos[c]++;
      sorted_row[dest] = static_cast<int>(rows[k]);
      sorted_orig_idx[dest] = k;
    }

    colptrs_.resize(n_vars_ + 1);
    for (int c = 0; c <= n_vars_; c++)
      colptrs_[c] = static_cast<sunindextype>(col_count[c]);

    rowvals_.resize(nnz_total);
    data_indices_.resize(nnz_total);
    for (int i = 0; i < nnz_total; i++) {
      rowvals_[i] = static_cast<sunindextype>(sorted_row[i]);
      data_indices_[i] = sorted_orig_idx[i];
    }
    nnz_ = nnz_total;

    J_ = SUNSparseMatrix(n_vars_, n_vars_, nnz_, CSC_MAT, sunctx_);
    sunindextype* jp = SUNSparseMatrix_IndexPointers(J_);
    sunindextype* jr = SUNSparseMatrix_IndexValues(J_);
    for (int i = 0; i <= n_vars_; i++) jp[i] = colptrs_[i];
    for (int i = 0; i < nnz_; i++) jr[i] = rowvals_[i];

    LS_ = SUNLinSol_KLU(delta_nvec_, J_, sunctx_);
  }

  void BuildDenseResources() {
    const auto& rows = jac_cf_.get_row();
    const auto& cols = jac_cf_.get_col();
    int nnz_total = static_cast<int>(jac_cf_.nnz_out());

    // Build CSC structure for dense scatter
    std::vector<int> col_count(n_vars_ + 1, 0);
    for (int k = 0; k < nnz_total; k++)
      col_count[static_cast<int>(cols[k]) + 1]++;
    for (int c = 1; c <= n_vars_; c++)
      col_count[c] += col_count[c - 1];

    std::vector<int> sorted_row(nnz_total);
    std::vector<int> sorted_orig_idx(nnz_total);
    std::vector<int> pos(col_count);
    for (int k = 0; k < nnz_total; k++) {
      int c = static_cast<int>(cols[k]);
      int dest = pos[c]++;
      sorted_row[dest] = static_cast<int>(rows[k]);
      sorted_orig_idx[dest] = k;
    }

    colptrs_.resize(n_vars_ + 1);
    for (int c = 0; c <= n_vars_; c++)
      colptrs_[c] = static_cast<sunindextype>(col_count[c]);

    rowvals_.resize(nnz_total);
    data_indices_.resize(nnz_total);
    for (int i = 0; i < nnz_total; i++) {
      rowvals_[i] = static_cast<sunindextype>(sorted_row[i]);
      data_indices_[i] = sorted_orig_idx[i];
    }
    nnz_ = nnz_total;

    J_ = SUNDenseMatrix(n_vars_, n_vars_, sunctx_);
    LS_ = SUNLinSol_Dense(delta_nvec_, J_, sunctx_);
  }

  CasadiFunction res_cf_;
  CasadiFunction jac_cf_;
  int n_vars_;
  bool use_sparse_;

  SUNContext sunctx_;
  SUNMatrix J_;
  SUNLinearSolver LS_;
  N_Vector res_nvec_;
  N_Vector delta_nvec_;

  int nnz_ = 0;
  std::vector<sunindextype> colptrs_;
  std::vector<sunindextype> rowvals_;
  std::vector<int> data_indices_;
  std::vector<sunrealtype> jac_buf_;
  std::vector<sunrealtype> inputs_;
};

/**
 * @brief Standalone Newton solver exposed to Python via pybind11.
 *
 * Owns a StandaloneAlgebraicSystem and NonlinearSolver. No IDA dependency.
 * Solves F(t, y, inputs) = 0 for y given an initial guess.
 */
class StandaloneNewtonSolver {
public:
  StandaloneNewtonSolver(
    casadi::Function residual_fn,
    casadi::Function jacobian_fn,
    const std::vector<sunrealtype>& atol,
    sunrealtype rtol,
    sunrealtype step_tol,
    int max_iter,
    int max_backtracks,
    sunrealtype epsNewt,
    bool use_sparse)
    : system_(residual_fn, jacobian_fn, use_sparse),
      solver_(system_, system_.n_vars(), atol.data(), rtol, step_tol,
              max_iter, max_backtracks, epsNewt),
      n_vars_(system_.n_vars()),
      y_work_(n_vars_)
  {}

  /**
   * @brief Solve F(t, y, inputs) = 0 starting from y0.
   * @return (success, y_solution) with zero-copy numpy output.
   */
  std::pair<bool, np_array> solve(
    sunrealtype t,
    const np_array& y0_np,
    const np_array& inputs_np)
  {
    auto y0 = y0_np.unchecked<1>();
    auto inp = inputs_np.unchecked<1>();
    system_.set_inputs(inp.data(0), static_cast<int>(inp.size()));
    std::memcpy(y_work_.data(), y0.data(0), n_vars_ * sizeof(sunrealtype));

    NonlinearResult result = solver_.solve_single(t, y_work_.data());

    np_array out({n_vars_});
    std::memcpy(out.mutable_data(), y_work_.data(),
                n_vars_ * sizeof(sunrealtype));
    return {nonlinear_success(result), std::move(out)};
  }

  /**
   * @brief Batch solve over multiple time points in a single C++ call.
   *
   * Each solve reuses the previous solution as the initial guess.
   * Stops early on the first failure.
   *
   * @return (all_success, y_matrix) where y_matrix has shape (n_vars, n_times).
   */
  std::pair<bool, py::array_t<sunrealtype, py::array::f_style>> solve_batch(
    const np_array& t_eval_np,
    const np_array& y0_alg_np,
    const np_array& inputs_np)
  {
    auto t_eval = t_eval_np.unchecked<1>();
    auto y0 = y0_alg_np.unchecked<1>();
    auto inp = inputs_np.unchecked<1>();

    int n_times = static_cast<int>(t_eval.size());
    system_.set_inputs(inp.data(0), static_cast<int>(inp.size()));
    std::memcpy(y_work_.data(), y0.data(0), n_vars_ * sizeof(sunrealtype));

    py::array_t<sunrealtype, py::array::f_style> out({n_vars_, n_times});
    sunrealtype* out_data = out.mutable_data();

    for (int i = 0; i < n_times; i++) {
      NonlinearResult result = solver_.solve_single(t_eval(i), y_work_.data());
      if (!nonlinear_success(result))
        return {false, std::move(out)};
      std::memcpy(out_data + i * n_vars_, y_work_.data(),
                  n_vars_ * sizeof(sunrealtype));
    }

    return {true, std::move(out)};
  }

private:
  StandaloneAlgebraicSystem system_;
  NonlinearSolver solver_;
  int n_vars_;
  std::vector<sunrealtype> y_work_;
};

#endif // PYBAMM_STANDALONE_NEWTON_SOLVER_HPP
