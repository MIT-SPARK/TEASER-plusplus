get_filename_component(TEASERPP_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(CMakeFindDependencyMacro)

find_dependency(Eigen3 3.3 REQUIRED)

include("${TEASERPP_CMAKE_DIR}/teaserppTargets.cmake")
