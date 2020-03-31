cmake_minimum_required(VERSION 3.10)
project(teaserpp_example)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(teaserpp REQUIRED)

# Change this line to include your own executable file
add_executable(cpp_ply cpp_ply.cc)

# Link to teaserpp & Eigen3
target_link_libraries(cpp_ply Eigen3::Eigen teaserpp::teaser_registration teaserpp::teaser_io)

# Copy the data files to build directory
file(COPY ../example_data/
        DESTINATION ./example_data/
        FILES_MATCHING
        PATTERN *.ply)
