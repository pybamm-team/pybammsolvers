import os
import subprocess
import argparse
import platform
import shutil
from os.path import join, isfile
from multiprocessing import cpu_count
import pathlib

DEFAULT_INSTALL_DIR = str(pathlib.Path(__file__).parent.resolve() / ".idaklu")


def build_solvers():
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count())
    parser = argparse.ArgumentParser(
        description="Compile and install Sundials and SuiteSparse."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force installation even if libraries are already found. This will overwrite the pre-existing files.",
    )
    args = parser.parse_args()

    if args.force:
        print(
            "The '--force' option is activated: installation will be forced, ignoring any existing libraries."
        )
        safe_remove_dir(pathlib.Path("build_sundials"))
        sundials_found, suitesparse_found = False, False
    else:
        sundials_found, suitesparse_found = check_libraries_installed()

    if not suitesparse_found:
        install_suitesparse()
    if not sundials_found:
        install_sundials()


def check_libraries_installed():
    lib_dirs = [DEFAULT_INSTALL_DIR]

    sundials_files = [
        "libsundials_idas",
        "libsundials_sunlinsolklu",
        "libsundials_sunlinsoldense",
        "libsundials_sunlinsolspbcgs",
        "libsundials_sunlinsollapackdense",
        "libsundials_sunmatrixsparse",
        "libsundials_nvecserial",
        "libsundials_nvecopenmp",
    ]

    sundials_lib_found = find_library_files("Sundials", lib_dirs, sundials_files)

    suitesparse_files = [
        "libsuitesparseconfig",
        "libklu",
        "libamd",
        "libcolamd",
        "libbtf",
    ]

    suitesparse_lib_found = find_library_files(
        "SuiteSparse", lib_dirs, suitesparse_files
    )

    return sundials_lib_found, suitesparse_lib_found


def find_library_files(library_name, lib_dirs, file_names):
    if platform.system() == "Linux":
        file_names = [file + ".so" for file in file_names]
    elif platform.system() == "Darwin":
        file_names = [file + ".dylib" for file in file_names]
    else:
        file_names = [file + ".dll" for file in file_names]
    lib_found = True

    for lib_file in file_names:
        file_found = False
        for lib_dir in lib_dirs:
            if isfile(join(lib_dir, "lib", lib_file)):
                print(f"{lib_file} found in {lib_dir}.")
                file_found = True
                break
        if not file_found:
            print(
                f"{lib_file} not found. Proceeding with {library_name} library installation."
            )
            lib_found = False
            break
    return lib_found


def install_sundials():
    KLU_INCLUDE_DIR = os.path.join(DEFAULT_INSTALL_DIR, "include", "suitesparse")
    KLU_LIBRARY_DIR = os.path.join(DEFAULT_INSTALL_DIR, "lib")
    cmake_args = [
        "-DENABLE_LAPACK=ON",
        "-DSUNDIALS_INDEX_SIZE=32",
        "-DEXAMPLES_ENABLE_C=OFF",
        "-DEXAMPLES_ENABLE_CXX=OFF",
        "-DEXAMPLES_INSTALL=OFF",
        "-DENABLE_KLU=ON",
        "-DENABLE_OPENMP=ON",
        f"-DKLU_INCLUDE_DIR={KLU_INCLUDE_DIR}",
        f"-DKLU_LIBRARY_DIR={KLU_LIBRARY_DIR}",
        "-DCMAKE_INSTALL_PREFIX=" + DEFAULT_INSTALL_DIR,
        "-DCMAKE_INSTALL_NAME_DIR=" + KLU_LIBRARY_DIR,
    ]

    # try to find OpenMP on Mac
    if platform.system() == "Darwin":
        # flags to find OpenMP on Mac
        if platform.processor() == "arm":
            OpenMP_C_FLAGS = (
                "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
            )
            OpenMP_C_LIB_NAMES = "omp"
            OpenMP_omp_LIBRARY = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
        elif platform.processor() == "i386":
            OpenMP_C_FLAGS = "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include"
            OpenMP_C_LIB_NAMES = "omp"
            OpenMP_omp_LIBRARY = "/usr/local/opt/libomp/lib/libomp.dylib"
        else:
            raise NotImplementedError(
                f"Unsupported processor architecture: {platform.processor()}. "
                "Only 'arm' and 'i386' architectures are supported."
            )

        # Don't pass the following args to CMake when building wheels. We set a custom
        # OpenMP installation for macOS wheels in the wheel build script.
        # This is because we can't use Homebrew's OpenMP dylib due to the wheel
        # repair process, where Homebrew binaries are not built for distribution and
        # break MACOSX_DEPLOYMENT_TARGET. We use a custom OpenMP binary as described
        # in CIBW_BEFORE_ALL in the wheel builder CI job.
        # Check for CI environment variable to determine if we are building a wheel
        if os.environ.get("CIBUILDWHEEL") != "1":
            cmake_args += [
                "-DOpenMP_C_FLAGS=" + OpenMP_C_FLAGS,
                "-DOpenMP_C_LIB_NAMES=" + OpenMP_C_LIB_NAMES,
                "-DOpenMP_omp_LIBRARY=" + OpenMP_omp_LIBRARY,
            ]

    build_dir = pathlib.Path("build_sundials")
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    sundials_src = "../sundials"
    subprocess.run(["cmake", sundials_src, *cmake_args], cwd=build_dir, check=True)
    make_cmd = ["make", f"-j{cpu_count()}", "install"]
    subprocess.run(make_cmd, cwd=build_dir, check=True)


def install_suitesparse():
    klu_dependencies = ["SuiteSparse_config", "AMD", "COLAMD", "BTF", "KLU"]
    suitesparse_src = pathlib.Path("SuiteSparse")
    make_cmd = [
        "make",
        "library",
    ]
    install_cmd = [
        "make",
        f"-j{cpu_count()}",
        "install",
    ]
    # Set CMAKE_OPTIONS as environment variables to pass to the GNU Make command
    cmake_options = ""
    env = os.environ.copy()
    for libdir in klu_dependencies:
        build_dir = os.path.join(suitesparse_src, libdir)
        # We want to ensure that libsuitesparseconfig.dylib is not repeated in
        # multiple paths at the time of wheel repair. Therefore, it should not be
        # built with an RPATH since it is copied to the install prefix.
        if libdir == "SuiteSparse_config":
            # if in CI, set RPATH to the install directory for SuiteSparse_config
            # dylibs to find libomp.dylib when repairing the wheel
            if os.environ.get("CIBUILDWHEEL") == "1":
                cmake_options = (
                    f" -DCMAKE_INSTALL_PREFIX={DEFAULT_INSTALL_DIR}"
                    f" -DCMAKE_INSTALL_RPATH={DEFAULT_INSTALL_DIR}/lib"
                )
            else:
                cmake_options = f"-DCMAKE_INSTALL_PREFIX={DEFAULT_INSTALL_DIR}"
        else:
            # For AMD, COLAMD, BTF and KLU; do not set a BUILD RPATH but use an
            # INSTALL RPATH in order to ensure that the dynamic libraries are found
            # at runtime just once. Otherwise, delocate complains about multiple
            # references to the SuiteSparse_config dynamic library (auditwheel does not).
            cmake_options = (
                f" -DCMAKE_INSTALL_PREFIX={DEFAULT_INSTALL_DIR}"
                f" -DCMAKE_INSTALL_RPATH={DEFAULT_INSTALL_DIR}/lib"
                f" -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE"
                f" -DCMAKE_BUILD_WITH_INSTALL_RPATH=FALSE"
            )
        vcpkg_dir = os.environ.get("VCPKG_ROOT_DIR", None)
        triplet = os.environ.get("VCPKG_DEFAULT_TRIPLET", None)
        if vcpkg_dir:
            cmake_options += f" -DBLAS_ROOT={vcpkg_dir}/vcpkg_installed/{triplet}"
        env["CMAKE_OPTIONS"] = cmake_options
        subprocess.run(make_cmd, cwd=build_dir, env=env, shell=True, check=True)
        subprocess.run(install_cmd, cwd=build_dir, check=True)


def safe_remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def check_build_tools():
    try:
        subprocess.run(["make", "--version"])
    except OSError:
        raise RuntimeError("Make must be installed.") from None
    try:
        subprocess.run(["cmake", "--version"])
    except OSError:
        raise RuntimeError("CMake must be installed.") from None


if __name__ == "__main__":
    check_build_tools()
    build_solvers()
