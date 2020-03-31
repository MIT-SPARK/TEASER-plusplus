cmake_minimum_required(VERSION 3.10)
project(teaserpp_example)

set (CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(teaserpp REQUIRED)

# Change this line to include your own executable file
add_executable(cpp_fpfh cpp_fpfh.cpp)

# Link to teaserpp & Eigen3
target_link_libraries(cpp_fpfh Eigen3::Eigen teaserpp::teaser_registration teaserp::teaser_io)