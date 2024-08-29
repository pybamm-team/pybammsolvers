import pybamm, pybammsolvers 
import numpy as np

def tests_run():
    assert True

def test_changing_grid():
    model = pybamm.lithium_ion.SPM()

    # load parameter values and geometry
    geometry = model.default_geometry
    param = model.default_parameter_values

    # Process parameters
    param.process_model(model)
    param.process_geometry(geometry)

    # Calculate time for each solver and each number of grid points
    t_eval = np.linspace(0, 3600, 100)
    for npts in [100, 200]:
        # discretise
        var_pts = {
            spatial_var: npts for spatial_var in ["x_n", "x_s", "x_p", "r_n", "r_p"]
        }
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        model_disc = disc.process_model(model, inplace=False)
        solver = pybammsolvers.IDAKLUSolver()

        # solve
        solver.solve(model_disc, t_eval)

def test_interpolation():
        model = pybamm.BaseModel()
        u1 = pybamm.Variable("u1")
        u2 = pybamm.Variable("u2")
        u3 = pybamm.Variable("u3")
        v = pybamm.Variable("v")
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b", expected_size=2)
        model.rhs = {u1: a * v, u2: pybamm.Index(b, 0), u3: pybamm.Index(b, 1)}
        model.algebraic = {v: 1 - v}
        model.initial_conditions = {u1: 0, u2: 0, u3: 0, v: 1}

        disc = pybamm.Discretisation()
        model_disc = disc.process_model(model, inplace=False)

        a_value = 0.1
        b_value = np.array([[0.2], [0.3]])
        inputs = {"a": a_value, "b": b_value}

        # Calculate time for each solver and each number of grid points
        t0 = 0
        tf = 3600
        t_eval_dense = np.linspace(t0, tf, 1000)
        t_eval_sparse = [t0, tf]

        t_interp_dense = np.linspace(t0, tf, 800)
        t_interp_sparse = [t0, tf]
        solver = pybammsolvers.IDAKLUSolver()

        # solve
        # 1. dense t_eval + adaptive time stepping
        sol1 = solver.solve(model_disc, t_eval_dense, inputs=inputs)
        np.testing.assert_array_less(len(t_eval_dense), len(sol1.t))

        # 2. sparse t_eval + adaptive time stepping
        sol2 = solver.solve(model_disc, t_eval_sparse, inputs=inputs)
        np.testing.assert_array_less(len(sol2.t), len(sol1.t))

        # 3. dense t_eval + dense t_interp
        sol3 = solver.solve(
            model_disc, t_eval_dense, t_interp=t_interp_dense, inputs=inputs
        )
        t_combined = np.concatenate((sol3.t, t_interp_dense))
        t_combined = np.unique(t_combined)
        t_combined.sort()
        np.testing.assert_array_almost_equal(sol3.t, t_combined)

        # 4. sparse t_eval + sparse t_interp
        sol4 = solver.solve(
            model_disc, t_eval_sparse, t_interp=t_interp_sparse, inputs=inputs
        )
        np.testing.assert_array_almost_equal(sol4.t, np.array([t0, tf]))

        sols = [sol1, sol2, sol3, sol4]
        for sol in sols:
            # test that y[0] = to true solution
            true_solution = a_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[0], true_solution)

            # test that y[1:3] = to true solution
            true_solution = b_value * sol.t
            np.testing.assert_array_almost_equal(sol.y[1:3], true_solution)


def test_model_events():
    for form in ["casadi", "iree"]:
        if (form == "jax" or form == "iree") and not pybammsolvers.have_jax():
            continue
        if (form == "iree") and not pybammsolvers.have_iree():
            continue
        if form == "casadi":
            root_method = "casadi"
        else:
            root_method = "lm"
        # Create model
        model = pybamm.BaseModel()
        model.convert_to_format = "jax" if form == "iree" else form
        var = pybamm.Variable("var")
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}

        # create discretisation
        disc = pybamm.Discretisation()
        model_disc = disc.process_model(model, inplace=False)
        # Solve
        solver = pybammsolvers.IDAKLUSolver(
            rtol=1e-8,
            atol=1e-8,
            root_method=root_method,
            options={"jax_evaluator": "iree"} if form == "iree" else {},
        )

        if model.convert_to_format == "casadi" or (
            model.convert_to_format == "jax"
            and solver._options["jax_evaluator"] == "iree"
        ):
            t_interp = np.linspace(0, 1, 100)
            t_eval = np.array([t_interp[0], t_interp[-1]])
        else:
            t_eval = np.linspace(0, 1, 100)
            t_interp = t_eval

        solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
        np.testing.assert_array_equal(
            solution.t, t_interp, err_msg=f"Failed for form {form}"
        )
        np.testing.assert_array_almost_equal(
            solution.y[0],
            np.exp(0.1 * solution.t),
            decimal=5,
            err_msg=f"Failed for form {form}",
        )

        # Check invalid atol type raises an error
        with self.assertRaises(pybamm.SolverError):
            solver._check_atol_type({"key": "value"}, [])

        # enforce events that won't be triggered
        model.events = [pybamm.Event("an event", var + 1)]
        model_disc = disc.process_model(model, inplace=False)
        solver = pybamm.IDAKLUSolver(
            rtol=1e-8,
            atol=1e-8,
            root_method=root_method,
            options={"jax_evaluator": "iree"} if form == "iree" else {},
        )
        solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
        np.testing.assert_array_equal(solution.t, t_interp)
        np.testing.assert_array_almost_equal(
            solution.y[0],
            np.exp(0.1 * solution.t),
            decimal=5,
            err_msg=f"Failed for form {form}",
        )

        # enforce events that will be triggered
        model.events = [pybamm.Event("an event", 1.01 - var)]
        model_disc = disc.process_model(model, inplace=False)
        solver = pybamm.IDAKLUSolver(
            rtol=1e-8,
            atol=1e-8,
            root_method=root_method,
            options={"jax_evaluator": "iree"} if form == "iree" else {},
        )
        solution = solver.solve(model_disc, t_eval, t_interp=t_interp)
        self.assertLess(len(solution.t), len(t_interp))
        np.testing.assert_array_almost_equal(
            solution.y[0],
            np.exp(0.1 * solution.t),
            decimal=5,
            err_msg=f"Failed for form {form}",
        )

        # bigger dae model with multiple events
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        var1 = pybamm.Variable("var1", domain=whole_cell)
        var2 = pybamm.Variable("var2", domain=whole_cell)
        model.rhs = {var1: 0.1 * var1}
        model.algebraic = {var2: 2 * var1 - var2}
        model.initial_conditions = {var1: 1, var2: 2}
        model.events = [
            pybamm.Event("var1 = 1.5", pybamm.min(1.5 - var1)),
            pybamm.Event("var2 = 2.5", pybamm.min(2.5 - var2)),
        ]
        disc = get_discretisation_for_testing()
        disc.process_model(model)

        solver = pybammsolvers.IDAKLUSolver(
            rtol=1e-8,
            atol=1e-8,
            root_method=root_method,
            options={"jax_evaluator": "iree"} if form == "iree" else {},
        )
        t_eval = np.array([0, 5])
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_less(solution.y[0, :-1], 1.5)
        np.testing.assert_array_less(solution.y[-1, :-1], 2.5)
        np.testing.assert_equal(solution.t_event[0], solution.t[-1])
        np.testing.assert_array_equal(solution.y_event[:, 0], solution.y[:, -1])
        np.testing.assert_array_almost_equal(
            solution.y[0],
            np.exp(0.1 * solution.t),
            decimal=5,
            err_msg=f"Failed for form {form}",
        )
        np.testing.assert_array_almost_equal(
            solution.y[-1],
            2 * np.exp(0.1 * solution.t),
            decimal=5,
            err_msg=f"Failed for form {form}",
        )
