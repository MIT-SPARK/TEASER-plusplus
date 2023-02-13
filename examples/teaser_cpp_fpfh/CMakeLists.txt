cmake_minimum_required(VERSION 3.10)
project(teaser_cpp_fpfh)

set (CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(teaserpp REQUIRED)
find_package(PCL 1.9 REQUIRED COMPONENTS common features kdtree)
find_package(OpenMP)

# Change this line to include your own executable file
add_executable(quatro_cpp_fpfh quatro_cpp_fpfh.cc)

# Link to teaserpp & Eigen3
target_link_libraries(quatro_cpp_fpfh Eigen3::Eigen teaserpp::teaser_registration teaserpp::teaser_features
        teaserpp::teaser_io ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

# Change this line to include your own executable file
add_executable(teaser_cpp_fpfh teaser_cpp_fpfh.cc)

# Link to teaserpp & Eigen3
target_link_libraries(teaser_cpp_fpfh Eigen3::Eigen teaserpp::teaser_registration teaserpp::teaser_features
        teaserpp::teaser_io ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

# Copy the data files to build directory
file(COPY ../example_data/
        DESTINATION ./example_data/
        FILES_MATCHING
        PATTERN *.ply)
