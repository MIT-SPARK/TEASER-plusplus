cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(teaser_python_bindings)

pybind11_add_module(teaserpp_python teaserpp_python/teaserpp_python.cc)

message(STATUS "Python Interpreter Version: ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
if (NOT (PYTHON_VERSION_MAJOR EQUAL 2 AND PYTHON_VERSION_MINOR EQUAL 7))
    # Hack: VTK used in PCL might add /usr/include/python2.7 to all targets'
    # INCLUDE_DIRECTORIES properties. We need to remove it.
    get_target_property(TEASERPY_NEW_INCLUDE_DIRS teaserpp_python INTERFACE_INCLUDE_DIRECTORIES)
    list(FILTER TEASERPY_NEW_INCLUDE_DIRS EXCLUDE REGEX ".*python2.7$")
    set_target_properties(teaserpp_python
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TEASERPY_NEW_INCLUDE_DIRS}")
endif ()

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
    INSTALL_RPATH "$ORIGIN/../teaserpp_python.libs/lib;$ORIGIN/../../teaser/"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# copy package __init__.py file
configure_file(teaserpp_python/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/teaserpp_python/__init__.py
        )

configure_file(teaserpp_python/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/teaserpp_python/_teaserpp.pyi
        )

# copy setup.py file
configure_file(setup.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/setup.py
        )

file(COPY .
        DESTINATION .
        FILES_MATCHING
        PATTERN *.py)

if (DEFINED SKBUILD)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/teaserpp/
            DESTINATION "../teaserpp"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "*.pyi"
            PATTERN "*.so"
    )
    install(TARGETS teaserpp_python DESTINATION "../teaserpp_python")
endif ()
