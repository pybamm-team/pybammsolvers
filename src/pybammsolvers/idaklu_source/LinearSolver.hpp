#ifndef PYBAMM_LINEAR_SOLVER_HPP
#define PYBAMM_LINEAR_SOLVER_HPP

#include "common.hpp"
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>
#include <functional>

/**
 * @brief Direct and iterative linear solver wrapping SUNDIALS.
 *
 * Supports all SUNDIALS linear solvers: KLU (sparse), Dense, Band,
 * SPBCGS, SPFGMR, SPGMR, SPTFQMR.
 *
 * Two ownership modes:
 *  - Owns resources: creates SUNContext, SUNMatrix, SUNLinearSolver, N_Vectors.
 *  - Borrows resources: wraps existing IDA resources, does not free them.
 *
 * Direct solvers: factorize(values) fills the matrix and factors it,
 * then solve(b, x) uses the factored form. Zero allocations in hotpath.
 *
 * Iterative solvers: set_atimes() to provide J*v callback, then solve()
 * performs the iterative solve. factorize() throws.
 */
class LinearSolver {
public:
  using ATimesFn = std::function<int(const sunrealtype* v, sunrealtype* Jv)>;

  /**
   * @brief Construct with explicit CSC sparsity (owned resources).
   * @param n       System dimension.
   * @param nnz     Number of nonzeros.
   * @param colptrs CSC column pointers (length n+1).
   * @param rowvals CSC row indices (length nnz).
   * @param options Dict with "jacobian", "linear_solver", etc.
   */
  LinearSolver(int n, int nnz,
               const int64_t* colptrs, const int64_t* rowvals,
               py::dict options);

  /**
   * @brief Construct from scipy.sparse matrix (owned resources).
   * Converts to CSC internally.
   */
  LinearSolver(py::object scipy_sparse, py::dict options);

  /**
   * @brief Construct borrowing IDA's resources (no ownership of LS/J/sunctx).
   * Creates own N_Vectors for zero-copy solve via raw pointers.
   */
  LinearSolver(int n, SUNContext sunctx,
               SUNLinearSolver ls, SUNMatrix j);

  ~LinearSolver();

  LinearSolver(const LinearSolver&) = delete;
  LinearSolver& operator=(const LinearSolver&) = delete;

  // Properties
  int n() const { return n_; }
  int nnz() const { return nnz_; }
  bool can_factorize() const { return can_factorize_; }
  bool owns_resources() const { return owns_resources_; }

  // -- Direct solver path --

  // Fill matrix with values and factorize. Throws if !can_factorize().
  // For sparse: values has nnz() elements.
  // For dense: values has n()*n() elements (column-major).
  void factorize(const sunrealtype* values);
  void factorize(np_array values_np);

  // Solve A*x = b using existing factorization (or iterative solve).
  // b and x may alias. No allocations.
  void solve(const sunrealtype* b, sunrealtype* x);

  // Python-facing: writes into out if provided.
  np_array solve(np_array b_np, py::object out = py::none());

  // Batched solve: A*X = B, B is (n, k), X is (n, k). Same factorization.
  void solve(const sunrealtype* B, sunrealtype* X, int k);
  np_array solve_batched(np_array B_np, py::object out = py::none());

  // -- Iterative solver path --
  void set_atimes(ATimesFn fn);

  // Internal accessors for IDA integration (borrowed mode)
  SUNMatrix J_ptr() const { return J_; }
  SUNLinearSolver LS_ptr() const { return LS_; }
  N_Vector get_b_nvec() const { return b_nvec_; }
  N_Vector get_x_nvec() const { return x_nvec_; }

private:
  void InitOwned(int n, int nnz,
                 const int64_t* colptrs, const int64_t* rowvals,
                 const std::string& jacobian_type,
                 const std::string& linear_solver_type,
                 int linsol_max_iterations,
                 int jac_bandwidth_lower, int jac_bandwidth_upper);

  static int atimes_callback(void* data, N_Vector v, N_Vector z);

  int n_;
  int nnz_;
  bool owns_resources_;
  bool can_factorize_;
  bool is_sparse_;
  bool is_dense_;
  bool is_banded_;

  SUNContext sunctx_;
  SUNMatrix J_;
  SUNLinearSolver LS_;

  // Pre-allocated N_Vectors for SUNLinSolSolve (owned mode only).
  // We swap their internal data pointers on each call -- zero allocation.
  N_Vector b_nvec_;
  N_Vector x_nvec_;

  ATimesFn atimes_fn_;
};

#include "LinearSolver.inl"

#endif // PYBAMM_LINEAR_SOLVER_HPP
