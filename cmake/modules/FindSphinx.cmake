# Sphinx configuration
# Credit: https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
find_program(SPHINX_EXECUTABLE
        NAMES sphinx-build
        DOC "sphinx-build command")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx
        "Failed to find sphinx-build executable"
        SPHINX_EXECUTABLE)
