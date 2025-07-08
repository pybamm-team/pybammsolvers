import nox
import os
import sys
from pathlib import Path


nox.options.default_venv_backend = "virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["pre-commit", "idaklu-requires", "unit"]

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


@nox.session(name="pre-commit")
def lint(session):
    """Check all files against the defined pre-commit hooks."""
    session.install("pre-commit", silent=False)
    session.run("pre-commit", "run", "--all-files")


@nox.session(name="idaklu-requires")
def run_pybamm_requires(session):
    """Download, compile, and install the build-time requirements for Linux and macOS.
    Supports --install-dir for custom installation paths and --force to force installation."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.error("nox -s idaklu-requires is only available on Linux & macOS.")


@nox.session(name="unit")
def run_unit(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("casadi==3.6.7", silent=False)
    session.install(".[dev]", silent=False)
    session.run("pytest", "tests")
