cmake_minimum_required(VERSION 3.16)
project(teaserpp VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/" ${CMAKE_MODULE_PATH})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check build types
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE
            STRING "Choose the type of build." FORCE)
endif ()

if (DEFINED SKBUILD)
    message(STATUS "Building with Scikit-Build")
endif ()

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_TEASER_FPFH "Build TEASER++ wrappers for PCL FPFH estimation." OFF)
option(BUILD_MATLAB_BINDINGS "Build MATLAB bindings" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_DOC "Build documentation" ON)
option(BUILD_WITH_MARCH_NATIVE "Build with flag march=native" OFF)
option(ENABLE_MKL "Try to use Eigen with MKL" OFF)
option(ENABLE_DIAGNOSTIC_PRINT "Enable printing of diagnostic messages" ON)

if (ENABLE_DIAGNOSTIC_PRINT)
    message(STATUS "Enable printing of diagnostic messages.")
    add_definitions(-DTEASER_DIAG_PRINT)
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
    message(STATUS "TEASER++ Python binding will be built.")
    add_subdirectory(python)
endif ()

# export targets
export(TARGETS ${TEASERPP_EXPORTED_TARGETS} FILE teaserpp-exports.cmake)

install(FILES cmake/teaserppConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/teaserpp
        )
