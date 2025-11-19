# This module is adapted from that in CADET (`<https://github.com/modsim/CADET)>`_):

# .. cmake_module::

#    Find SUNDIALS, the SUite of Nonlinear and DIfferential/ALgebraic equation Solvers.
#
#    The module looks for the following sundials components
#
#    * sundials_ida
#    * sundials_sunlinsolklu
#    * sundials_sunlinsoldense
#    * sundials_sunlinsollapackdense
#    * sundials_sunmatrix_sparse
#    * sundials_nvecserial
#
#    To provide the module with a hint about where to find your SUNDIALS installation,
#    you can set the environment variable :code:`SUNDIALS_ROOT`. The FindSUNDIALS module will
#    then look in this path when searching for SUNDIALS paths and libraries.
#    This behavior is defined in CMake >= 3.12, see policy CMP0074.
#    It is replicated for older versions by adding the :code:`SUNDIALS_ROOT` variable to the
#    :code:`PATHS` entry.
#
#    This module will define the following variables:
#    :code:`SUNDIALS_INCLUDE_DIR` - Location of the SUNDIALS includes
#    :code:`SUNDIALS_LIBRARIES` - Required libraries for all requested components

# List of the valid SUNDIALS components

# find the SUNDIALS include directories
find_path(SUNDIALS_INCLUDE_DIR
  NAMES
    idas/idas.h
    sundials/sundials_math.h
    sundials/sundials_types.h
    sunlinsol/sunlinsol_klu.h
    sunlinsol/sunlinsol_dense.h
    sunlinsol/sunlinsol_spbcgs.h
    sunlinsol/sunlinsol_lapackdense.h
    sunmatrix/sunmatrix_sparse.h
  PATH_SUFFIXES
    include
  PATHS
    ${SUNDIALS_ROOT}
  )

set(SUNDIALS_WANT_COMPONENTS
  sundials_core
  sundials_idas
  sundials_sunlinsolklu
  sundials_sunlinsoldense
  sundials_sunlinsolspbcgs
  sundials_sunlinsollapackdense
  sundials_sunmatrixsparse
  sundials_nvecserial
  sundials_nvecopenmp
  )

# find the SUNDIALS libraries
foreach(LIB ${SUNDIALS_WANT_COMPONENTS})
    if (UNIX AND SUNDIALS_PREFER_STATIC_LIBRARIES)
        # According to bug 1643 on the CMake bug tracker, this is the
        # preferred method for searching for a static library.
        # See http://www.cmake.org/Bug/view.php?id=1643.  We search
        # first for the full static library name, but fall back to a
        # generic search on the name if the static search fails.
        set(THIS_LIBRARY_SEARCH lib${LIB}.a ${LIB})
    elseif(WIN32)
        # On Windows, try both libsundials_* and sundials_* naming conventions
        # CMake's find_library will automatically add .lib extension
        set(THIS_LIBRARY_SEARCH lib${LIB} ${LIB})
        # Verbose debug output for sundials_core on Windows
        if(LIB STREQUAL "sundials_core")
            message(STATUS "")
            message(STATUS "================================================================")
            message(STATUS "SUNDIALS sundials_core search (Windows) - VERBOSE DEBUG")
            message(STATUS "================================================================")
            message(STATUS "Library component: ${LIB}")
            message(STATUS "Searching for library names: ${THIS_LIBRARY_SEARCH}")
            message(STATUS "  (CMake will automatically try .lib extension)")
            message(STATUS "")
            message(STATUS "SUNDIALS_ROOT variable: ${SUNDIALS_ROOT}")
            message(STATUS "SUNDIALS_ROOT (expanded): ${SUNDIALS_ROOT}")
            if(SUNDIALS_ROOT)
                get_filename_component(SUNDIALS_ROOT_ABS "${SUNDIALS_ROOT}" ABSOLUTE)
                message(STATUS "SUNDIALS_ROOT (absolute): ${SUNDIALS_ROOT_ABS}")
            endif()
            message(STATUS "")
            message(STATUS "Explicit search paths (from PATHS):")
            if(SUNDIALS_ROOT)
                message(STATUS "  [1] ${SUNDIALS_ROOT}")
                message(STATUS "      Checking: ${SUNDIALS_ROOT}/lib")
                message(STATUS "      Checking: ${SUNDIALS_ROOT}/Lib")
            else()
                message(STATUS "  [1] (SUNDIALS_ROOT not set - will search system paths)")
            endif()
            message(STATUS "")
            message(STATUS "PATH_SUFFIXES that will be appended:")
            message(STATUS "  - lib")
            message(STATUS "  - Lib")
            message(STATUS "")
            message(STATUS "Full list of paths CMake will search:")
            if(SUNDIALS_ROOT)
                set(SEARCH_PATH_1 "${SUNDIALS_ROOT}/lib")
                set(SEARCH_PATH_2 "${SUNDIALS_ROOT}/Lib")
                message(STATUS "  [1] ${SEARCH_PATH_1}")
                if(EXISTS "${SEARCH_PATH_1}")
                    message(STATUS "      → EXISTS")
                    file(GLOB LIB_FILES_1 "${SEARCH_PATH_1}/*sundials_core*")
                    if(LIB_FILES_1)
                        message(STATUS "      → Found files matching '*sundials_core*':")
                        foreach(FILE ${LIB_FILES_1})
                            message(STATUS "        - ${FILE}")
                        endforeach()
                    else()
                        message(STATUS "      → No files matching '*sundials_core*' found")
                    endif()
                else()
                    message(STATUS "      → DOES NOT EXIST")
                endif()
                message(STATUS "  [2] ${SEARCH_PATH_2}")
                if(EXISTS "${SEARCH_PATH_2}")
                    message(STATUS "      → EXISTS")
                    file(GLOB LIB_FILES_2 "${SEARCH_PATH_2}/*sundials_core*")
                    if(LIB_FILES_2)
                        message(STATUS "      → Found files matching '*sundials_core*':")
                        foreach(FILE ${LIB_FILES_2})
                            message(STATUS "        - ${FILE}")
                        endforeach()
                    else()
                        message(STATUS "      → No files matching '*sundials_core*' found")
                    endif()
                else()
                    message(STATUS "      → DOES NOT EXIST")
                endif()
            else()
                message(STATUS "  (SUNDIALS_ROOT not set - CMake will search system paths)")
                message(STATUS "  System library paths:")
                get_property(CMAKE_SYSTEM_PREFIX_PATH_VAR GLOBAL PROPERTY CMAKE_SYSTEM_PREFIX_PATH)
                message(STATUS "    CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}")
            endif()
            message(STATUS "")
        endif()
    else()
        set(THIS_LIBRARY_SEARCH ${LIB})
    endif()

    find_library(SUNDIALS_${LIB}_LIBRARY
        NAMES ${THIS_LIBRARY_SEARCH}
        PATH_SUFFIXES
            lib
            Lib
	PATHS
	    ${SUNDIALS_ROOT}
    )

    set(SUNDIALS_${LIB}_FOUND FALSE)
    if (SUNDIALS_${LIB}_LIBRARY)
      list(APPEND SUNDIALS_LIBRARIES ${SUNDIALS_${LIB}_LIBRARY})
      set(SUNDIALS_${LIB}_FOUND TRUE)
    endif()
    mark_as_advanced(SUNDIALS_${LIB}_LIBRARY)
    
    # Verbose debug output for sundials_core on Windows
    if(WIN32 AND LIB STREQUAL "sundials_core")
        message(STATUS "Search result:")
        if(SUNDIALS_${LIB}_FOUND)
            message(STATUS "  ✓ SUNDIALS sundials_core FOUND")
            message(STATUS "  Library path: ${SUNDIALS_${LIB}_LIBRARY}")
            get_filename_component(LIB_DIR "${SUNDIALS_${LIB}_LIBRARY}" DIRECTORY)
            get_filename_component(LIB_NAME "${SUNDIALS_${LIB}_LIBRARY}" NAME)
            message(STATUS "  Library directory: ${LIB_DIR}")
            message(STATUS "  Library filename: ${LIB_NAME}")
            message(STATUS "  Will be linked: YES")
            message(STATUS "  Status: SUCCESS - Library will be available for linking")
        else()
            message(STATUS "  ✗ SUNDIALS sundials_core NOT FOUND")
            message(STATUS "  Searched for names: ${THIS_LIBRARY_SEARCH}")
            message(STATUS "  Searched in paths:")
            if(SUNDIALS_ROOT)
                message(STATUS "    - ${SUNDIALS_ROOT}/lib")
                message(STATUS "    - ${SUNDIALS_ROOT}/Lib")
            else()
                message(STATUS "    - System library paths (SUNDIALS_ROOT not set)")
            endif()
            message(STATUS "  Will be linked: NO")
            message(STATUS "  Impact: SUNContext_ClearErrHandlers may not be available at link time")
            message(STATUS "  Note: This is OK if SUNDIALS_VERSION_MAJOR < 7 (function won't be called)")
            message(STATUS "")
            message(STATUS "Troubleshooting tips:")
            message(STATUS "  1. Check if sundials_core.lib exists in ${SUNDIALS_ROOT}/lib or ${SUNDIALS_ROOT}/Lib")
            message(STATUS "  2. Verify SUNDIALS_ROOT is set correctly: ${SUNDIALS_ROOT}")
            message(STATUS "  3. For SUNDIALS < 7, sundials_core may not exist (this is expected)")
            message(STATUS "  4. Check if library is named differently (e.g., libsundials_core.lib)")
        endif()
        message(STATUS "================================================================")
        message(STATUS "")
    endif()
endforeach()

mark_as_advanced(
    SUNDIALS_LIBRARIES
    SUNDIALS_INCLUDE_DIR
)

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  "SUNDIALS"
  FOUND_VAR SUNDIALS_FOUND
  REQUIRED_VARS SUNDIALS_INCLUDE_DIR SUNDIALS_LIBRARIES
  HANDLE_COMPONENTS
)
