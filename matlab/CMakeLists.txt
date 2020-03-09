cmake_minimum_required(VERSION 3.10)

project(teaser_matlab_bindings)

find_package(Matlab COMPONENTS MX_LIBRARY)

if (Matlab_FOUND)
    message(STATUS "MATLAB root directory found: ${Matlab_ROOT_DIR}.")
    matlab_add_mex(
            NAME teaser_mex
            OUTPUT_NAME teaser_solve_mex
            SRC teaser_mex.cc
            LINK_TO Eigen3::Eigen teaser_registration
    )
    set_target_properties(teaser_mex PROPERTIES COMPILE_FLAGS "-fvisibility=default")
    # copy MATLAB .m files to binary directory
    file(COPY .
            DESTINATION .
            FILES_MATCHING
            PATTERN *.m)
else ()
    message(WARNING "MATLAB root directory not found. Failed to build MATLAB bindings.")
    set(BUILD_MATLAB_BINDINGS OFF)
endif ()