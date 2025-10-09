import nox
import os
import sys
from pathlib import Path


nox.options.default_venv_backend = "virtualenv"
nox.options.reuse_existing_virtualenvs = True
if sys.platform != "win32":
    nox.options.sessions = ["idaklu-requires", "unit"]
else:
    nox.options.sessions = ["unit"]

homedir = Path(__file__)
PYBAMM_ENV = {
    "LD_LIBRARY_PATH": f"{homedir}/.idaklu/lib",
    "PYTHONIOENCODING": "utf-8",
    "MPLBACKEND": "Agg",
    # Expression evaluators (...EXPR_CASADI cannot be fully disabled at this time)
    "PYBAMM_IDAKLU_EXPR_CASADI": os.getenv("PYBAMM_IDAKLU_EXPR_CASADI", "ON"),
}
VENV_DIR = Path("./venv").resolve()


def set_environment_variables(env_dict, session):
    """
    Sets environment variables for a nox Session object.

    Parameters
    -----------
        session : nox.Session
            The session to set the environment variables for.
        env_dict : dict
            A dictionary of environment variable names and values.

    """
    for key, value in env_dict.items():
        session.env[key] = value


@nox.session(name="idaklu-requires")
def run_pybamm_requires(session):
    """Download, compile, and install the build-time requirements for Linux and macOS.
    Supports --install-dir for custom installation paths and --force to force installation."""
    set_environment_variables(PYBAMM_ENV, session=session)
    if sys.platform != "win32":
        session.run("python", "install_KLU_Sundials.py", *session.posargs)
    else:
        session.error("nox -s idaklu-requires is only available on Linux & macOS.")


@nox.session(name="unit")
def run_unit(session):
    """Run the full test suite."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("casadi==3.6.7", silent=False)
    session.install(".[dev]", silent=False)
    session.run("pytest", "tests", *session.posargs)


@nox.session(name="unit-fast")
def run_unit_fast(session):
    """Run fast tests only (excluding slow and integration tests)."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("casadi==3.6.7", silent=False)
    session.install(".[dev]", silent=False)
    session.run(
        "pytest", "tests", "-m", "not slow and not integration", *session.posargs
    )


@nox.session(name="unit-integration")
def run_unit_integration(session):
    """Run integration tests only."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("casadi==3.6.7", silent=False)
    session.install(".[dev]", silent=False)
    session.run("pytest", "tests", "-m", "integration", "-v", *session.posargs)


@nox.session(name="unit-performance")
def run_unit_performance(session):
    """Run performance tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("casadi==3.6.7", silent=False)
    session.install(".[dev]", silent=False)
    # Install optional performance testing dependencies
    session.install("psutil", silent=False)
    session.run("pytest", "tests/test_performance.py", "-v", *session.posargs)


@nox.session(name="coverage")
def run_coverage(session):
    """Run tests with coverage reporting."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("casadi==3.6.7", silent=False)
    session.install(".[dev]", silent=False)
    session.install("pytest-cov", silent=False)
    session.run(
        "pytest",
        "tests",
        "--cov=pybammsolvers",
        "--cov-report=html",
        "--cov-report=term-missing",
        *session.posargs,
    )
