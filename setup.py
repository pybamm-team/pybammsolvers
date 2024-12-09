import os
import sys
import subprocess
from multiprocessing import cpu_count
from pathlib import Path
from platform import system

from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
from setuptools.command.bdist_wheel import bdist_wheel


default_lib_dir = (
    "" if system() == "Windows" else str(Path(__file__).parent.resolve() / ".idaklu")
)

# ---------- set environment variables for vcpkg on Windows ----------------------------


def set_vcpkg_environment_variables():
    if not os.getenv("VCPKG_ROOT_DIR"):
        raise OSError("Environment variable 'VCPKG_ROOT_DIR' is undefined.")
    if not os.getenv("VCPKG_DEFAULT_TRIPLET"):
        raise OSError("Environment variable 'VCPKG_DEFAULT_TRIPLET' is undefined.")
    if not os.getenv("VCPKG_FEATURE_FLAGS"):
        raise OSError("Environment variable 'VCPKG_FEATURE_FLAGS' is undefined.")
    return (
        os.getenv("VCPKG_ROOT_DIR"),
        os.getenv("VCPKG_DEFAULT_TRIPLET"),
        os.getenv("VCPKG_FEATURE_FLAGS"),
    )


# ---------- CMakeBuild class (custom build_ext for IDAKLU target) ---------------------


class CMakeBuild(build_ext):
    user_options = [
        *build_ext.user_options,
        ("suitesparse-root=", None, "suitesparse source location"),
        ("sundials-root=", None, "sundials source location"),
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.suitesparse_root = None
        self.sundials_root = None

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Determine the calling command to get the
        # undefined options from.
        # If build_ext was called directly then this
        # doesn't matter.
        try:
            self.get_finalized_command("install", create=0)
            calling_cmd = "install"
        except AttributeError:
            calling_cmd = "bdist_wheel"
        self.set_undefined_options(
            calling_cmd,
            ("suitesparse_root", "suitesparse_root"),
            ("sundials_root", "sundials_root"),
        )
        if not self.suitesparse_root:
            self.suitesparse_root = os.path.join(default_lib_dir)
        if not self.sundials_root:
            self.sundials_root = os.path.join(default_lib_dir)

    def get_build_directory(self):
        # setuptools outputs object files in directory self.build_temp
        # (typically build/temp.*). This is our CMake build directory.
        # On Windows, setuptools is too smart and appends "Release" or
        # "Debug" to self.build_temp. So in this case we want the
        # build directory to be the parent directory.
        if system() == "Windows":
            return Path(self.build_temp).parents[0]
        return self.build_temp

    def run(self):
        if not self.extensions:
            return

        # Build in parallel wherever possible
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count())

        if system() == "Windows":
            use_python_casadi = False
        else:
            use_python_casadi = True

        build_type = os.getenv("PYBAMM_CPP_BUILD_TYPE", "RELEASE")
        idaklu_expr_casadi = os.getenv("PYBAMM_IDAKLU_EXPR_CASADI", "ON")
        idaklu_expr_iree = os.getenv("PYBAMM_IDAKLU_EXPR_IREE", "OFF")
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DUSE_PYTHON_CASADI={}".format("TRUE" if use_python_casadi else "FALSE"),
            f"-DPYBAMM_IDAKLU_EXPR_CASADI={idaklu_expr_casadi}",
            f"-DPYBAMM_IDAKLU_EXPR_IREE={idaklu_expr_iree}",
        ]
        if self.suitesparse_root:
            cmake_args.append(
                f"-DSuiteSparse_ROOT={os.path.abspath(self.suitesparse_root)}"
            )
        if self.sundials_root:
            cmake_args.append(f"-DSUNDIALS_ROOT={os.path.abspath(self.sundials_root)}")

        build_dir = self.get_build_directory()
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        # The CMakeError.log file is generated by cmake is the configure step
        # encounters error. In the following the existence of this file is used
        # to determine whether the cmake configure step went smoothly.
        # So must make sure this file does not remain from a previous failed build.
        if os.path.isfile(os.path.join(build_dir, "CMakeError.log")):
            os.remove(os.path.join(build_dir, "CMakeError.log"))

        # ---------- configuration for vcpkg on Windows ----------------------------------------

        build_env = os.environ
        if os.getenv("PYBAMMSOLVERS_USE_VCPKG"):
            (
                vcpkg_root_dir,
                vcpkg_default_triplet,
                vcpkg_feature_flags,
            ) = set_vcpkg_environment_variables()
            build_env["vcpkg_root_dir"] = vcpkg_root_dir
            build_env["vcpkg_default_triplet"] = vcpkg_default_triplet
            build_env["vcpkg_feature_flags"] = vcpkg_feature_flags

        # ---------- Run CMake and build IDAKLU module -----------------------------------------

        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print("-" * 10, "Running CMake for IDAKLU solver", "-" * 40)
        subprocess.run(
            ["cmake", cmake_list_dir, *cmake_args],
            cwd=build_dir,
            env=build_env,
            check=True,
        )

        if os.path.isfile(os.path.join(build_dir, "CMakeError.log")):
            msg = (
                "cmake configuration steps encountered errors, and the IDAKLU module"
                " could not be built. Make sure dependencies are correctly "
                "installed. See "
                "https://docs.pybamm.org/en/latest/source/user_guide/installation/install-from-source.html"
            )
            raise RuntimeError(msg)
        else:
            print("-" * 10, "Building IDAKLU module", "-" * 40)
            subprocess.run(
                ["cmake", "--build", ".", "--config", "Release"],
                cwd=build_dir,
                env=build_env,
                check=True,
            )

            # Move from build temp to final position
            for ext in self.extensions:
                self.move_output(ext)

    def move_output(self, ext):
        # Copy built module to dist/ directory
        build_temp = Path(self.build_temp).resolve()
        # Get destination location
        # self.get_ext_fullpath(ext.name) -->
        # build/lib.linux-x86_64-3.5/idaklu.cpython-37m-x86_64-linux-gnu.so
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / os.path.basename(self.get_ext_filename(ext.name))
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


# ---------- end of CMake steps --------------------------------------------------------


class CustomInstall(install):
    """A custom installation command to add 2 build options"""

    user_options = [
        *install.user_options,
        ("suitesparse-root=", None, "suitesparse source location"),
        ("sundials-root=", None, "sundials source location"),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.suitesparse_root = None
        self.sundials_root = None

    def finalize_options(self):
        install.finalize_options(self)
        if not self.suitesparse_root:
            self.suitesparse_root = default_lib_dir
        if not self.sundials_root:
            self.sundials_root = default_lib_dir

    def run(self):
        install.run(self)


# ---------- Custom class for building wheels ------------------------------------------


class PyBaMMWheel(bdist_wheel):
    """A custom installation command to add 2 build options"""

    user_options = [
        *bdist_wheel.user_options,
        ("suitesparse-root=", None, "suitesparse source location"),
        ("sundials-root=", None, "sundials source location"),
    ]

    def initialize_options(self):
        bdist_wheel.initialize_options(self)
        self.suitesparse_root = None
        self.sundials_root = None

    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        if not self.suitesparse_root:
            self.suitesparse_root = default_lib_dir
        if not self.sundials_root:
            self.sundials_root = default_lib_dir

    def run(self):
        bdist_wheel.run(self)


ext_modules = [
    Extension(
        name="pybammsolvers.idaklu",
        # The sources list should mirror the list in CMakeLists.txt
        sources=[
            "src/pybammsolvers/idaklu_source/Expressions/Expressions.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/Base/Expression.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/Base/ExpressionSet.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/Base/ExpressionTypes.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/Base/ExpressionSparsity.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/Casadi/CasadiFunctions.cpp",
            "src/pybammsolvers/idaklu_source/Expressions/Casadi/CasadiFunctions.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/IREE/IREEBaseFunction.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/IREE/IREEFunction.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/IREE/IREEFunctions.cpp",
            "src/pybammsolvers/idaklu_source/Expressions/IREE/IREEFunctions.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/IREE/iree_jit.cpp",
            "src/pybammsolvers/idaklu_source/Expressions/IREE/iree_jit.hpp",
            "src/pybammsolvers/idaklu_source/Expressions/IREE/ModuleParser.cpp",
            "src/pybammsolvers/idaklu_source/Expressions/IREE/ModuleParser.hpp",
            "src/pybammsolvers/idaklu_source/idaklu_solver.hpp",
            "src/pybammsolvers/idaklu_source/IDAKLUSolver.cpp",
            "src/pybammsolvers/idaklu_source/IDAKLUSolver.hpp",
            "src/pybammsolvers/idaklu_source/IDAKLUSolverGroup.cpp",
            "src/pybammsolvers/idaklu_source/IDAKLUSolverGroup.hpp",
            "src/pybammsolvers/idaklu_source/IDAKLUSolverOpenMP.inl",
            "src/pybammsolvers/idaklu_source/IDAKLUSolverOpenMP.hpp",
            "src/pybammsolvers/idaklu_source/IDAKLUSolverOpenMP_solvers.cpp",
            "src/pybammsolvers/idaklu_source/IDAKLUSolverOpenMP_solvers.hpp",
            "src/pybammsolvers/idaklu_source/sundials_functions.inl",
            "src/pybammsolvers/idaklu_source/sundials_functions.hpp",
            "src/pybammsolvers/idaklu_source/IdakluJax.cpp",
            "src/pybammsolvers/idaklu_source/IdakluJax.hpp",
            "src/pybammsolvers/idaklu_source/common.hpp",
            "src/pybammsolvers/idaklu_source/common.cpp",
            "src/pybammsolvers/idaklu_source/Solution.cpp",
            "src/pybammsolvers/idaklu_source/Solution.hpp",
            "src/pybammsolvers/idaklu_source/SolutionData.cpp",
            "src/pybammsolvers/idaklu_source/SolutionData.hpp",
            "src/pybammsolvers/idaklu_source/observe.cpp",
            "src/pybammsolvers/idaklu_source/observe.hpp",
            "src/pybammsolvers/idaklu_source/Options.hpp",
            "src/pybammsolvers/idaklu_source/Options.cpp",
            "src/pybammsolvers/idaklu.cpp",
        ],
    )
]

# Project metadata was moved to pyproject.toml (which is read by pip). However, custom
# build commands and setuptools extension modules are still defined here.
setup(
    # silence "Package would be ignored" warnings
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": CMakeBuild,
        "bdist_wheel": PyBaMMWheel,
        "install": CustomInstall,
    },
)
