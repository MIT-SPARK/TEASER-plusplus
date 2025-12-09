project(teaser_test)
include(GoogleTest)

set(TEST_SRCS
        main.cc
        utils-test.cc
        io-test.cc
        geometry-test.cc
        tls-test.cc
        scale-solver-test.cc
        rotation-solver-test.cc
        translation-solver-test.cc
        registration-test.cc
        graph-test.cc
        certification-test.cc
        linalg-test.cc)
set(TEST_LINK_LIBRARIES
        Eigen3::Eigen
        ${TEASERPP_BLAS_LAPACK_LIBS}
        gtest
        gmock
        teaser_io
        teaser_registration
        test_tools
        pmc)

# Add feature tests if TEASER_FPFH is enabled
if (BUILD_TEASER_FPFH)
    include_directories(${PCL_INCLUDE_DIRS})
    list(APPEND TEST_SRCS feature-test.cc
            matcher-test.cc)
    list(APPEND TEST_LINK_LIBRARIES teaser_features ${PCL_LIBRARIES})
endif ()

# Executable for running unit tests
add_executable(all_tests ${TEST_SRCS})
target_link_libraries(all_tests ${TEST_LINK_LIBRARIES})
target_include_directories(all_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../teaser/src)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(all_tests OpenMP::OpenMP_CXX)
endif()

gtest_add_tests(TARGET all_tests
        TEST_LIST allTests)
set_tests_properties(${allTests} PROPERTIES TIMEOUT 120)

# Copy test data files to binary directory
file(COPY .
        DESTINATION .
        FILES_MATCHING
        PATTERN *.ply
        PATTERN *.label
        PATTERN *.pcd
        PATTERN *.csv
        PATTERN *.txt)
