project(teaserpp_doc)

set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        ${CMAKE_CURRENT_SOURCE_DIR}/conf.py
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY .
        DESTINATION .
        FILES_MATCHING
        PATTERN *.rst
        )

# We use TEASERPP_SOURCE_DIR environmental variable as a way to
# control location of teaser source files without using CMake's configure_file,
# so that we can use both in source build and out-of-source build for documentation.
add_custom_target(doc ALL
        COMMAND TEASERPP_SOURCE_DIR=${teaserpp_SOURCE_DIR}/teaser doxygen
        WORKING_DIRECTORY ${teaserpp_BINARY_DIR}/doc)

# Sphinx
find_package(Sphinx)

if (Sphinx_FOUND)
    message(STATUS "Found Sphinx. Will build Sphinx HTML files.")
    set(SPHINX_SOURCE ${teaserpp_BINARY_DIR}/doc)
    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx_build)
    message(STATUS "Sphinx build location: ${CMAKE_CURRENT_BINARY_DIR}/sphinx_build")

    add_custom_target(sphinx ALL
            COMMAND
            ${SPHINX_EXECUTABLE} -b html
            ${SPHINX_SOURCE} ${SPHINX_BUILD}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating documentation with Sphinx")
endif ()