#pragma once

#include <sunmatrix/sunmatrix_band.h>

// ────────────────────── Static ATimes wrapper ──────────────────────

inline int LinearSolver::atimes_callback(void* data, N_Vector v, N_Vector z) {
  auto* self = static_cast<LinearSolver*>(data);
  return self->atimes_fn_(N_VGetArrayPointer(v), N_VGetArrayPointer(z));
}

// ────────────────────── Shared init for owned resources ──────────────────────

inline void LinearSolver::InitOwned(
  int n, int nnz,
  const int64_t* colptrs, const int64_t* rowvals,
  const std::string& jacobian_type,
  const std::string& linear_solver_type,
  int linsol_max_iterations,
  int jac_bandwidth_lower, int jac_bandwidth_upper
) {
  n_ = n;
  owns_resources_ = true;

  SUNContext_Create(SUN_COMM_NULL, &sunctx_);

  // Determine solver type flags
  is_sparse_ = (jacobian_type == "sparse" && linear_solver_type == "SUNLinSol_KLU");
  is_dense_ = (jacobian_type == "dense" || jacobian_type == "none");
  is_banded_ = (jacobian_type == "banded");

  bool is_iterative = (linear_solver_type == "SUNLinSol_SPBCGS" ||
                       linear_solver_type == "SUNLinSol_SPFGMR" ||
                       linear_solver_type == "SUNLinSol_SPGMR" ||
                       linear_solver_type == "SUNLinSol_SPTFQMR");

  can_factorize_ = !is_iterative;

  // Create N_Vectors (pre-allocated, data pointers swapped on each solve)
  b_nvec_ = N_VNew_Serial(n_, sunctx_);
  x_nvec_ = N_VNew_Serial(n_, sunctx_);

  // Create matrix and linear solver
  if (is_sparse_) {
    nnz_ = nnz;
    J_ = SUNSparseMatrix(n_, n_, nnz_, CSC_MAT, sunctx_);
    LS_ = SUNLinSol_KLU(b_nvec_, J_, sunctx_);

    // Copy sparsity pattern into the SUNMatrix (fixed for lifetime)
    sunindextype* j_colptrs = SUNSparseMatrix_IndexPointers(J_);
    sunindextype* j_rowvals = SUNSparseMatrix_IndexValues(J_);
    for (int i = 0; i <= n_; i++) j_colptrs[i] = colptrs[i];
    for (int i = 0; i < nnz_; i++) j_rowvals[i] = rowvals[i];
  } else if (is_banded_) {
    nnz_ = nnz;
    J_ = SUNBandMatrix(n_, jac_bandwidth_upper, jac_bandwidth_lower, sunctx_);
    LS_ = SUNLinSol_Band(b_nvec_, J_, sunctx_);
  } else if (is_dense_) {
    nnz_ = 0;
    J_ = SUNDenseMatrix(n_, n_, sunctx_);
    LS_ = SUNLinSol_Dense(b_nvec_, J_, sunctx_);
  } else {
    // Iterative solvers
    nnz_ = 0;
    J_ = nullptr;
    if (linear_solver_type == "SUNLinSol_SPBCGS") {
      LS_ = SUNLinSol_SPBCGS(b_nvec_, SUN_PREC_NONE, linsol_max_iterations, sunctx_);
    } else if (linear_solver_type == "SUNLinSol_SPFGMR") {
      LS_ = SUNLinSol_SPFGMR(b_nvec_, SUN_PREC_NONE, linsol_max_iterations, sunctx_);
    } else if (linear_solver_type == "SUNLinSol_SPGMR") {
      LS_ = SUNLinSol_SPGMR(b_nvec_, SUN_PREC_NONE, linsol_max_iterations, sunctx_);
    } else if (linear_solver_type == "SUNLinSol_SPTFQMR") {
      LS_ = SUNLinSol_SPTFQMR(b_nvec_, SUN_PREC_NONE, linsol_max_iterations, sunctx_);
    } else {
      throw std::runtime_error(
        "LinearSolver: unknown linear_solver type '" + linear_solver_type + "'");
    }
  }

  SUNLinSolInitialize(LS_);
}

// ────────────────────── Constructor: explicit CSC sparsity ──────────────────────

inline LinearSolver::LinearSolver(
  int n, int nnz,
  const int64_t* colptrs, const int64_t* rowvals,
  py::dict options
) : sunctx_(nullptr), J_(nullptr), LS_(nullptr),
    b_nvec_(nullptr), x_nvec_(nullptr)
{
  std::string jacobian_type = options.contains("jacobian")
    ? options["jacobian"].cast<std::string>() : "sparse";
  std::string linear_solver_type = options.contains("linear_solver")
    ? options["linear_solver"].cast<std::string>() : "SUNLinSol_KLU";
  int linsol_max_iterations = options.contains("linsol_max_iterations")
    ? options["linsol_max_iterations"].cast<int>() : 100;
  int jac_bandwidth_lower = options.contains("jac_bandwidth_lower")
    ? options["jac_bandwidth_lower"].cast<int>() : 0;
  int jac_bandwidth_upper = options.contains("jac_bandwidth_upper")
    ? options["jac_bandwidth_upper"].cast<int>() : 0;

  InitOwned(n, nnz, colptrs, rowvals,
            jacobian_type, linear_solver_type,
            linsol_max_iterations,
            jac_bandwidth_lower, jac_bandwidth_upper);
}

// ────────────────────── Constructor: scipy sparse matrix ──────────────────────

inline LinearSolver::LinearSolver(
  py::object scipy_sparse, py::dict options
) : sunctx_(nullptr), J_(nullptr), LS_(nullptr),
    b_nvec_(nullptr), x_nvec_(nullptr)
{
  // Convert to CSC
  py::module_ sp = py::module_::import("scipy.sparse");
  py::object csc = sp.attr("csc_matrix")(scipy_sparse);

  py::array_t<int64_t> indptr = csc.attr("indptr").cast<py::array_t<int64_t>>();
  py::array_t<int64_t> indices = csc.attr("indices").cast<py::array_t<int64_t>>();
  int n = csc.attr("shape").cast<py::tuple>()[0].cast<int>();
  int nnz = indices.request().size;

  std::string jacobian_type = options.contains("jacobian")
    ? options["jacobian"].cast<std::string>() : "sparse";
  std::string linear_solver_type = options.contains("linear_solver")
    ? options["linear_solver"].cast<std::string>() : "SUNLinSol_KLU";
  int linsol_max_iterations = options.contains("linsol_max_iterations")
    ? options["linsol_max_iterations"].cast<int>() : 100;
  int jac_bandwidth_lower = options.contains("jac_bandwidth_lower")
    ? options["jac_bandwidth_lower"].cast<int>() : 0;
  int jac_bandwidth_upper = options.contains("jac_bandwidth_upper")
    ? options["jac_bandwidth_upper"].cast<int>() : 0;

  InitOwned(n, nnz,
            static_cast<const int64_t*>(indptr.request().ptr),
            static_cast<const int64_t*>(indices.request().ptr),
            jacobian_type, linear_solver_type,
            linsol_max_iterations,
            jac_bandwidth_lower, jac_bandwidth_upper);
}

// ────────────────────── Constructor: borrowed from IDA ──────────────────────

inline LinearSolver::LinearSolver(
  int n, SUNContext sunctx,
  SUNLinearSolver ls, SUNMatrix j
) : n_(n), nnz_(0), owns_resources_(false),
    sunctx_(sunctx), J_(j), LS_(ls),
    b_nvec_(nullptr), x_nvec_(nullptr)
{
  int ls_type = SUNLinSolGetType(ls);
  can_factorize_ = (ls_type == SUNLINEARSOLVER_DIRECT ||
                    ls_type == SUNLINEARSOLVER_MATRIX_ITERATIVE);

  is_sparse_ = (j != nullptr && SUNMatGetID(j) == SUNMATRIX_SPARSE);
  is_dense_ = (j != nullptr && SUNMatGetID(j) == SUNMATRIX_DENSE);
  is_banded_ = (j != nullptr && SUNMatGetID(j) == SUNMATRIX_BAND);

  if (is_sparse_) {
    nnz_ = SUNSparseMatrix_NNZ(j);
  }

  // Create N_Vectors for zero-copy solve path (owned by this LinearSolver,
  // even though LS/J/sunctx are borrowed)
  b_nvec_ = N_VNew_Serial(n_, sunctx_);
  x_nvec_ = N_VNew_Serial(n_, sunctx_);
}

// ────────────────────── Destructor ──────────────────────

inline LinearSolver::~LinearSolver() {
  // N_Vectors are always owned (created even in borrowed mode)
  if (b_nvec_) N_VDestroy(b_nvec_);
  if (x_nvec_) N_VDestroy(x_nvec_);
  if (owns_resources_) {
    if (LS_) SUNLinSolFree(LS_);
    if (J_) SUNMatDestroy(J_);
    if (sunctx_) SUNContext_Free(&sunctx_);
  }
}

// ────────────────────── factorize ──────────────────────

inline void LinearSolver::factorize(const sunrealtype* values) {
  if (!can_factorize_) {
    throw std::runtime_error(
      "LinearSolver::factorize() not supported for iterative solvers");
  }

  if (is_sparse_) {
    sunrealtype* data = SUNSparseMatrix_Data(J_);
    std::memcpy(data, values, nnz_ * sizeof(sunrealtype));
  } else if (is_banded_) {
    // Banded matrix: zero the matrix, then the caller must use
    // SM_COLUMN_ELEMENT_B to fill. For the LinearSolver standalone
    // case, banded is not used (only through IDA borrowed mode
    // where the Jacobian callback handles the fill directly).
    SUNMatZero(J_);
  } else if (is_dense_) {
    sunrealtype* data = SUNDenseMatrix_Data(J_);
    std::memcpy(data, values, n_ * n_ * sizeof(sunrealtype));
  }

  int flag = SUNLinSolSetup(LS_, J_);
  if (flag != 0) {
    throw std::runtime_error("LinearSolver::factorize: SUNLinSolSetup failed (flag=" +
                             std::to_string(flag) + ")");
  }
}

inline void LinearSolver::factorize(np_array values_np) {
  auto buf = values_np.request();
  factorize(static_cast<const sunrealtype*>(buf.ptr));
}

// ────────────────────── solve (C++ raw pointers) ──────────────────────

inline void LinearSolver::solve(const sunrealtype* b, sunrealtype* x) {
  // Zero-copy: point the N_Vector data at the caller's buffers
  NV_DATA_S(b_nvec_) = const_cast<sunrealtype*>(b);
  NV_DATA_S(x_nvec_) = x;

  int flag = SUNLinSolSolve(LS_, J_, x_nvec_, b_nvec_, SUN_RCONST(0.0));
  if (flag != 0) {
    throw std::runtime_error("LinearSolver::solve: SUNLinSolSolve failed (flag=" +
                             std::to_string(flag) + ")");
  }
}

// ────────────────────── solve (Python-facing) ──────────────────────

inline np_array LinearSolver::solve(np_array b_np, py::object out) {
  auto b_buf = b_np.request();
  if (b_buf.size != n_) {
    throw std::runtime_error("LinearSolver::solve: b has wrong size");
  }

  sunrealtype* x_ptr;
  np_array result;

  if (!out.is_none()) {
    result = out.cast<np_array>();
    auto out_buf = result.request();
    if (out_buf.size != n_) {
      throw std::runtime_error("LinearSolver::solve: out has wrong size");
    }
    x_ptr = static_cast<sunrealtype*>(out_buf.ptr);
  } else {
    result = np_array(n_);
    x_ptr = result.mutable_data();
  }

  solve(static_cast<const sunrealtype*>(b_buf.ptr), x_ptr);
  return result;
}

// ────────────────────── solve batched (C++ raw pointers) ──────────────────────

inline void LinearSolver::solve(const sunrealtype* B, sunrealtype* X, int k) {
  for (int col = 0; col < k; col++) {
    solve(B + col * n_, X + col * n_);
  }
}

// ────────────────────── solve_batched (Python-facing) ──────────────────────

inline np_array LinearSolver::solve_batched(np_array B_np, py::object out) {
  auto B_buf = B_np.request();
  if (B_buf.ndim != 2 || B_buf.shape[0] != n_) {
    throw std::runtime_error("LinearSolver::solve_batched: B must be (n, k)");
  }
  int k = static_cast<int>(B_buf.shape[1]);

  sunrealtype* X_ptr;
  np_array result;

  if (!out.is_none()) {
    result = out.cast<np_array>();
    auto out_buf = result.request();
    if (out_buf.ndim != 2 || out_buf.shape[0] != n_ || out_buf.shape[1] != k) {
      throw std::runtime_error("LinearSolver::solve_batched: out must be (n, k)");
    }
    X_ptr = static_cast<sunrealtype*>(out_buf.ptr);
  } else {
    result = np_array({n_, k});
    X_ptr = result.mutable_data();
  }

  const sunrealtype* B_ptr = static_cast<const sunrealtype*>(B_buf.ptr);
  solve(B_ptr, X_ptr, k);
  return result;
}

// ────────────────────── set_atimes ──────────────────────

inline void LinearSolver::set_atimes(ATimesFn fn) {
  atimes_fn_ = std::move(fn);
  SUNLinSolSetATimes(LS_, this, &LinearSolver::atimes_callback);
}
