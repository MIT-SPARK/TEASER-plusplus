project(teaserpp_doc)

set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

# Doxygen configuration
set(teaserpp_DOXY_PROJECT_NAME             "teaserpp")
set(teaserpp_DOXY_INPUT                    "\"${teaserpp_SOURCE_DIR}/teaser\" \"${teaserpp_SOURCE_DIR}/doc\"")

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
)

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
endif()