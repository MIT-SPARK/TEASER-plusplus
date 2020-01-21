project(teaser_all_test)
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
