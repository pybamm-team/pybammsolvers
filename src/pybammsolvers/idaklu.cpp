#include <vector>
#include <iostream>
#include <functional>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "idaklu_source/idaklu_solver.hpp"
#include "idaklu_source/observe.hpp"
#include "idaklu_source/IDAKLUSolverGroup.hpp"
#include "idaklu_source/IdakluJax.hpp"
#include "idaklu_source/common.hpp"
#include "idaklu_source/Expressions/Casadi/CasadiFunctions.hpp"
#include "idaklu_source/sundials_error_handler.hpp"
#include "idaklu_source/reduce.hpp"
#include "idaklu_source/LinearSolver.hpp"
#include "idaklu_source/NonlinearSolver.hpp"


casadi::Function generate_casadi_function(const std::string &data)
{
  return casadi::Function::deserialize(data);
}

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<np_array>);
PYBIND11_MAKE_OPAQUE(std::vector<np_array_realtype>);
PYBIND11_MAKE_OPAQUE(std::vector<Solution>);

PYBIND11_MODULE(idaklu, m)
{
  m.doc() = "sundials solvers"; // optional module docstring

  py::bind_vector<std::vector<np_array>>(m, "VectorNdArray");
  py::bind_vector<std::vector<np_array_realtype>>(m, "VectorRealtypeNdArray");
  py::bind_vector<std::vector<Solution>>(m, "VectorSolution");

  py::class_<IDAKLUSolverGroup>(m, "IDAKLUSolverGroup")
  .def("solve", &IDAKLUSolverGroup::solve,
    "perform a solve",
    py::arg("t_eval"),
    py::arg("t_interp"),
    py::arg("y0"),
    py::arg("yp0"),
    py::arg("inputs"),
    py::arg("logger") = py::none(),
    py::return_value_policy::take_ownership);

  m.def("create_casadi_solver_group", &create_idaklu_solver_group<CasadiFunctions>,
    "Create a group of casadi idaklu solver objects",
    py::arg("number_of_states"),
    py::arg("number_of_parameters"),
    py::arg("rhs_alg"),
    py::arg("jac_times_cjmass"),
    py::arg("jac_times_cjmass_colptrs"),
    py::arg("jac_times_cjmass_rowvals"),
    py::arg("jac_times_cjmass_nnz"),
    py::arg("jac_bandwidth_lower"),
    py::arg("jac_bandwidth_upper"),
    py::arg("jac_action"),
    py::arg("mass_action"),
    py::arg("sens"),
    py::arg("events"),
    py::arg("number_of_events"),
    py::arg("rhs_alg_id"),
    py::arg("atol"),
    py::arg("rtol"),
    py::arg("inputs"),
    py::arg("var_fcns"),
    py::arg("dvar_dy_fcns"),
    py::arg("dvar_dp_fcns"),
    py::arg("options"),
    py::arg("alg_res"),
    py::arg("alg_jac"),
    py::return_value_policy::take_ownership);

  m.def("observe", &observe,
    "Observe variables",
    py::arg("ts"),
    py::arg("ys"),
    py::arg("inputs"),
    py::arg("funcs"),
    py::arg("is_f_contiguous"),
    py::arg("shape"),
    py::return_value_policy::take_ownership);

  m.def("observe_hermite_interp", &observe_hermite_interp,
    "Observe and Hermite interpolate variables",
    py::arg("t_interp"),
    py::arg("ts"),
    py::arg("ys"),
    py::arg("yps"),
    py::arg("inputs"),
    py::arg("funcs"),
    py::arg("shape"),
    py::return_value_policy::take_ownership);

  m.def("generate_function", &generate_casadi_function,
    "Generate a casadi function",
    py::arg("string"),
    py::return_value_policy::take_ownership);

  m.def("sundials_error_message", &sundials_error_message,
    "Get a human-readable message for a SUNDIALS error code",
    py::arg("flag"),
    py::return_value_policy::copy);

  // IdakluJax interface routines
  py::class_<IdakluJax>(m, "IdakluJax")
    .def(
      "register_callback_eval",
      &IdakluJax::register_callback_eval,
      "Register a callback for function evaluation",
      py::arg("callback")
    )
    .def(
      "register_callback_jvp",
      &IdakluJax::register_callback_jvp,
      "Register a callback for JVP evaluation",
      py::arg("callback")
    )
    .def(
      "register_callback_vjp",
      &IdakluJax::register_callback_vjp,
      "Register a callback for the VJP evaluation",
      py::arg("callback")
    )
    .def(
      "register_callbacks",
      &IdakluJax::register_callbacks,
      "Register callbacks for function evaluation, JVP evaluation, and VJP evaluation",
      py::arg("callback_eval"),
      py::arg("callback_jvp"),
      py::arg("callback_vjp")
    )
    .def(
      "get_index",
      &IdakluJax::get_index,
      "Get the index of the JAXified instance"
    );
  m.def(
    "create_idaklu_jax",
    &create_idaklu_jax,
    "Create an idaklu jax object"
  );
  m.def(
    "registrations",
    &Registrations
  );

  m.def("reduce_knots", &reduce_knots,
    "Streaming knot reduction on multi-segment solution data",
    py::arg("ts"), py::arg("ys"), py::arg("yps"),
    py::arg("atols"), py::arg("t_evals"),
    py::arg("rtol"),
    py::arg("hermite_reduction_factor"));

  py::class_<casadi::Function>(m, "Function");

  py::class_<Solution>(m, "solution")
    .def_readwrite("t", &Solution::t)
    .def_readwrite("y", &Solution::y)
    .def_readwrite("yp", &Solution::yp)
    .def_readwrite("yS", &Solution::yS)
    .def_readwrite("ypS", &Solution::ypS)
    .def_readwrite("y_term", &Solution::y_term)
    .def_readwrite("flag", &Solution::flag)
    .def_readwrite("events_triggered", &Solution::events_triggered);

  py::class_<LinearSolver>(m, "LinearSolver")
    .def("factorize", static_cast<void (LinearSolver::*)(np_array)>(&LinearSolver::factorize),
      py::arg("values"))
    .def("solve", static_cast<np_array (LinearSolver::*)(np_array, py::object)>(&LinearSolver::solve),
      py::arg("b"), py::arg("out") = py::none())
    .def("solve_batched", &LinearSolver::solve_batched,
      py::arg("B"), py::arg("out") = py::none())
    .def_property_readonly("n", &LinearSolver::n)
    .def_property_readonly("nnz", &LinearSolver::nnz)
    .def_property_readonly("can_factorize", &LinearSolver::can_factorize);

  m.def("create_linear_solver",
    [](py::object scipy_sparse, py::dict options) {
      return new LinearSolver(scipy_sparse, options);
    },
    "Create a linear solver from a scipy sparse matrix",
    py::arg("matrix"),
    py::arg("options"),
    py::return_value_policy::take_ownership);

  py::class_<NonlinearSolver>(m, "NonlinearSolver")
    .def("solve",
      static_cast<py::object (NonlinearSolver::*)(np_array, np_array, np_array, py::object)>(&NonlinearSolver::solve),
      "Solve F(t, x, p) = 0 for each t in t_eval",
      py::arg("t_eval"),
      py::arg("x0"),
      py::arg("inputs"),
      py::arg("out") = py::none());

  m.def("create_nonlinear_solver",
    [](const casadi::Function &residual_fn,
       const casadi::Function &jacobian_fn,
       int n_vars,
       np_array atol,
       double rtol,
       double step_tol,
       int inputs_length,
       py::dict options) {
      return new NonlinearSolver(
        residual_fn, jacobian_fn, n_vars,
        atol, rtol, step_tol, inputs_length, options
      );
    },
    "Create a nonlinear solver for F(t, x, p) = 0",
    py::arg("residual"),
    py::arg("jacobian"),
    py::arg("n_vars"),
    py::arg("atol"),
    py::arg("rtol"),
    py::arg("step_tol"),
    py::arg("inputs"),
    py::arg("options"),
    py::return_value_policy::take_ownership);
}
