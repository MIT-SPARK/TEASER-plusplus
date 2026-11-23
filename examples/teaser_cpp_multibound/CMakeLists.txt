cmake_minimum_required(VERSION 3.16)
project(teaser_cpp_multibound LANGUAGES CXX)

find_package(OpenMP QUIET COMPONENTS CXX)
find_package(teaserpp CONFIG REQUIRED)

add_executable(teaser_cpp_multibound teaser_cpp_multibound.cc)
target_compile_features(teaser_cpp_multibound PRIVATE cxx_std_14)
target_link_libraries(teaser_cpp_multibound PRIVATE teaserpp::teaser_registration
                                                   teaserpp::teaser_io)
