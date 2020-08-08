project(teaserpp_doc)

set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

file(COPY
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(doc ALL
        COMMAND doxygen
        WORKING_DIRECTORY ${teaserpp_BINARY_DIR}/doc)

# Sphinx
find_package(Sphinx)

if (Sphinx_FOUND)
    message(STATUS "Found Sphinx. Will build Sphinx HTML files.")
    set(SPHINX_SOURCE ${teaserpp_SOURCE_DIR}/doc)
    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx_build)
    message(STATUS "Sphinx build location: ${CMAKE_CURRENT_BINARY_DIR}/sphinx_build")

    add_custom_target(Sphinx ALL
            COMMAND
            ${SPHINX_EXECUTABLE} -b html
            -Dbreathe_projects.teaserpp=${teaserpp_BINARY_DIR}/doc/xml
            ${SPHINX_SOURCE} ${SPHINX_BUILD}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating documentation with Sphinx")
endif ()