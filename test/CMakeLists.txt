project(teaser_all_test)

include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)
if (NOT googletest_POPULATED)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

include(GoogleTest)

add_subdirectory(test-tools)
add_subdirectory(teaser)
add_subdirectory(benchmark)

# Copy test data files to binary directory
file(COPY .
        DESTINATION .
        FILES_MATCHING
        PATTERN *.ply
        PATTERN *.label
        PATTERN *.pcd
        PATTERN *.csv
        PATTERN *.txt)
