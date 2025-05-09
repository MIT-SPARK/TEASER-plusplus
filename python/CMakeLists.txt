find_package(pybind11 REQUIRED)

pybind11_add_module(teaserpp_python teaserpp_python/teaserpp_python.cc)
target_link_libraries(teaserpp_python PUBLIC teaser_registration)

# fix for clang
# see: https://github.com/pybind/pybind11/issues/1818
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(teaserpp_python PUBLIC -fno-sized-deallocation)
endif ()

# make sure to output the build file to teaserpp_python folder
SET_TARGET_PROPERTIES(teaserpp_python
    PROPERTIES
    OUTPUT_NAME "_teaserpp"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/teaserpp_python"
)

if (DEFINED SKBUILD)
    install(TARGETS teaserpp_python DESTINATION "../teaserpp_python")
endif ()
