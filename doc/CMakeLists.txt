project(teaserpp_doc)

set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

set(teaserpp_DOXY_PROJECT_NAME             "teaserpp")
set(teaserpp_DOXY_INPUT                    "\"${teaserpp_SOURCE_DIR}/teaser\" \"${teaserpp_SOURCE_DIR}/doc\"")

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
)

add_custom_target(doc ALL
        COMMAND doxygen
        WORKING_DIRECTORY ${teaserpp_BINARY_DIR}/doc)
