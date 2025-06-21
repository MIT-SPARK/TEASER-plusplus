cmake_minimum_required(VERSION 3.10)
project(teaserpp VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/" ${CMAKE_MODULE_PATH})

# Check build types
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE
            STRING "Choose the type of build." FORCE)
endif ()

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_TEASER_FPFH "Build TEASER++ wrappers for PCL FPFH estimation." OFF)
option(BUILD_MATLAB_BINDINGS "Build MATLAB bindings" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_DOC "Build documentation" ON)
option(BUILD_WITH_MARCH_NATIVE "Build with flag march=native" OFF)
option(ENABLE_MKL "Try to use Eigen with MKL" OFF)
option(ENABLE_DIAGNOSTIC_PRINT "Enable printing of diagnostic messages" ON)

if (ENABLE_DIAGNOSTIC_PRINT)
    message(STATUS "Enable printing of diagnostic messages.")
    add_definitions(-DTEASER_DIAG_PRINT)
endif ()

# Cache Variables
if (NOT TEASERPP_PYTHON_VERSION)
    set(TEASERPP_PYTHON_VERSION "" CACHE STRING "Python version to use for TEASER++ bindings.")
endif ()

# Find dependencies
# Eigen3
find_package(Eigen3 3.2 QUIET REQUIRED NO_MODULE)
if (ENABLE_MKL)
    find_package(MKL)
    if (MKL_FOUND)
        message(STATUS "MKL found at: ${MKL_LIBRARIES}")
        include_directories(${MKL_INCLUDE_DIR})
        add_definitions(-DEIGEN_USE_MKL_ALL)
        list(APPEND TEASERPP_BLAS_LAPACK_LIBS ${MKL_LIBRARIES})
    else ()
        message(STATUS "MKL not found.")
    endif ()
endif ()

if (BUILD_TEASER_FPFH)
    # Boost
    find_package(Boost 1.58 QUIET REQUIRED)

    # Suppress CMake warnings
    # see here: https://github.com/PointCloudLibrary/pcl/issues/3680
    if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
        set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
    endif()
    find_package(PCL 1.8 QUIET REQUIRED COMPONENTS common io features kdtree)

    # check for -march=native in PCL compile definitions
    get_target_property(PCL_COMPILE_OPTIONS pcl_common INTERFACE_COMPILE_OPTIONS)
    string(FIND "${PCL_COMPILE_OPTIONS}" "-march=native" PCL_IS_NATIVE)
    if (NOT PCL_IS_NATIVE EQUAL -1)
        # Fix Eigen alignment conflicting with pcl issue
        # See: http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
        add_definitions(-DEIGEN_MAX_STATIC_ALIGN_BYTES=0)

        if (NOT BUILD_WITH_MARCH_NATIVE)
            message (STATUS "PCL built with -march=native. Enable -march=native for TEASER++ as well")
            set(BUILD_WITH_MARCH_NATIVE ON)
        endif()
    endif()
endif ()

# googletest
configure_file(cmake/GoogleTest.CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
        "${CMAKE_BINARY_DIR}/googletest-build" EXCLUDE_FROM_ALL)

# pmc (Parallel Maximum Clique)
configure_file(cmake/pmc.CMakeLists.txt.in pmc-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pmc-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pmc-download")
add_subdirectory("${CMAKE_BINARY_DIR}/pmc-src"
        "${CMAKE_BINARY_DIR}/pmc-build")

# tinyply
configure_file(cmake/tinyply.CMakeLists.txt.in tinyply-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tinyply-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tinyply-download")
add_subdirectory("${CMAKE_BINARY_DIR}/tinyply-src"
        "${CMAKE_BINARY_DIR}/tinyply-build")
target_include_directories(tinyply PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/tinyply-src/source>)

# spectra
configure_file(cmake/spectra.CMakeLists.txt.in spectra-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/spectra-download")
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/spectra-download")
set(SPECTRA_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/spectra-src/include")

# Building Targets
set(TEASERPP_ROOT ${CMAKE_CURRENT_LIST_DIR})
add_subdirectory(teaser)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

if (BUILD_DOC)
    add_subdirectory(doc EXCLUDE_FROM_ALL)
endif ()

if (BUILD_MATLAB_BINDINGS)
    message(STATUS "Will build MATLAB bindings.")
    add_subdirectory(matlab)
endif ()

if (BUILD_PYTHON_BINDINGS)
    set(PYBIND11_PYTHON_VERSION ${TEASERPP_PYTHON_VERSION})

    # download the pybind11 repo
    configure_file(cmake/pybind11.CMakeLists.txt.in pybind11-download/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pybind11-download")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pybind11-download")
    add_subdirectory("${CMAKE_BINARY_DIR}/pybind11-src"
            "${CMAKE_BINARY_DIR}/pybind11-build")

    message(STATUS "TEASER++ Python binding will be built.")
    add_subdirectory(python)
endif ()

# export targets
export(TARGETS ${TEASERPP_EXPORTED_TARGETS} FILE teaserpp-exports.cmake)

install(FILES cmake/teaserppConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/teaserpp
        )
