#ifndef PYBAMM_CREATE_IDAKLU_SOLVER_HPP
#define PYBAMM_CREATE_IDAKLU_SOLVER_HPP

#include "IDAKLUSolverOpenMP_solvers.hpp"
#include "IDAKLUSolverGroup.hpp"
#include <memory>

/**
 * Creates a concrete solver given a linear solver, as specified in
 * options_cpp.linear_solver.
 * @brief Create a concrete solver given a linear solver
 */
template<class ExprSet>
IDAKLUSolver *create_idaklu_solver(
  std::unique_ptr<ExprSet> functions,
  int number_of_parameters,
  const np_array_int &jac_times_cjmass_colptrs,
  const np_array_int &jac_times_cjmass_rowvals,
  const int jac_times_cjmass_nnz,
  const int jac_bandwidth_lower,
  const int jac_bandwidth_upper,
  const int number_of_events,
  np_array rhs_alg_id,
  np_array atol_np,
  double rel_tol,
  int inputs_length,
  SolverOptions solver_opts,
  SetupOptions setup_opts
) {

  IDAKLUSolver *idakluSolver = nullptr;

  // Instantiate solver class
  if (setup_opts.linear_solver == "SUNLinSol_Dense")
  {
    DEBUG("\tsetting SUNLinSol_Dense linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_Dense<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_KLU")
  {
    DEBUG("\tsetting SUNLinSol_KLU linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_KLU<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_Band")
  {
    DEBUG("\tsetting SUNLinSol_Band linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_Band<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_SPBCGS")
  {
    DEBUG("\tsetting SUNLinSol_SPBCGS_linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPBCGS<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_SPFGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPFGMR_linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPFGMR<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_SPGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPGMR<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }
  else if (setup_opts.linear_solver == "SUNLinSol_SPTFQMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPTFQMR<ExprSet>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      setup_opts,
      solver_opts
     );
  }

  if (idakluSolver == nullptr) {
    throw std::invalid_argument("Unsupported solver requested");
  }

  return idakluSolver;
}

/**
 * @brief Create a group of solvers using create_idaklu_solver
 */
template<class ExprSet>
IDAKLUSolverGroup *create_idaklu_solver_group(
  int number_of_states,
  int number_of_parameters,
  const typename ExprSet::BaseFunctionType &rhs_alg,
  const typename ExprSet::BaseFunctionType &jac_times_cjmass,
  const np_array_int &jac_times_cjmass_colptrs,
  const np_array_int &jac_times_cjmass_rowvals,
  const int jac_times_cjmass_nnz,
  const int jac_bandwidth_lower,
  const int jac_bandwidth_upper,
  const typename ExprSet::BaseFunctionType &jac_action,
  const typename ExprSet::BaseFunctionType &mass_action,
  const typename ExprSet::BaseFunctionType &sens,
  const typename ExprSet::BaseFunctionType &events,
  const int number_of_events,
  np_array rhs_alg_id,
  np_array atol_np,
  double rel_tol,
  int inputs_length,
  const std::vector<typename ExprSet::BaseFunctionType*>& var_fcns,
  const std::vector<typename ExprSet::BaseFunctionType*>& dvar_dy_fcns,
  const std::vector<typename ExprSet::BaseFunctionType*>& dvar_dp_fcns,
  py::dict py_opts
) {
  auto setup_opts = SetupOptions(py_opts);
  auto solver_opts = SolverOptions(py_opts);


  std::vector<std::unique_ptr<IDAKLUSolver>> solvers;
  for (int i = 0; i < setup_opts.num_solvers; i++) {
    // Note: we can't copy an ExprSet as it contains raw pointers to the functions
    // So we create it in the loop
    auto functions = std::make_unique<ExprSet>(
      rhs_alg,
      jac_times_cjmass,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      jac_times_cjmass_rowvals,
      jac_times_cjmass_colptrs,
      inputs_length,
      jac_action,
      mass_action,
      sens,
      events,
      number_of_states,
      number_of_events,
      number_of_parameters,
      var_fcns,
      dvar_dy_fcns,
      dvar_dp_fcns,
      setup_opts
    );
    solvers.emplace_back(
      std::unique_ptr<IDAKLUSolver>(
        create_idaklu_solver(
          std::move(functions),
          number_of_parameters,
          jac_times_cjmass_colptrs,
          jac_times_cjmass_rowvals,
          jac_times_cjmass_nnz,
          jac_bandwidth_lower,
          jac_bandwidth_upper,
          number_of_events,
          rhs_alg_id,
          atol_np,
          rel_tol,
          inputs_length,
          solver_opts,
          setup_opts
        )
      )
    );
  }

  return new IDAKLUSolverGroup(std::move(solvers), number_of_states, number_of_parameters);
}



#endif // PYBAMM_CREATE_IDAKLU_SOLVER_HPP
