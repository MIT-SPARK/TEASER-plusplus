project(teaser_test_tools)

add_library(test_tools INTERFACE)

target_link_libraries(test_tools INTERFACE
        Eigen3::Eigen
        teaser_registration)

target_include_directories(test_tools INTERFACE ./)