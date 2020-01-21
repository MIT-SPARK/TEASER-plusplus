project(teaser_doc)

set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

set(TEASER_DOXY_PROJECT_NAME             "TEASER")
set(TEASER_DOXY_INPUT                    "\"${TEASER_SOURCE_DIR}/teaser\" \"${TEASER_SOURCE_DIR}/doc\"")

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
)

add_custom_target(doc ALL
        COMMAND doxygen
        WORKING_DIRECTORY ${TEASER_BINARY_DIR}/doc)
