project(teaser_source)
include(GNUInstallDirs)

# teaser_io library
add_library(teaser_io SHARED src/ply_io.cc)
target_link_libraries(teaser_io PRIVATE tinyply)
target_include_directories(teaser_io PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS teaser_io
        EXPORT teaserpp-export
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
list(APPEND TEASERPP_EXPORTED_TARGETS teaser_io tinyply)
add_library(teaserpp::teaser_io ALIAS teaser_io)

# teaser_registration library
add_library(teaser_registration SHARED
        src/certification.cc
        src/registration.cc
        src/graph.cc
        )
target_link_libraries(teaser_registration
        PUBLIC Eigen3::Eigen
        PRIVATE pmc ${TEASERPP_BLAS_LAPACK_LIBS}
        )
target_include_directories(teaser_registration PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${SPECTRA_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(teaser_registration PUBLIC OpenMP::OpenMP_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

install(TARGETS teaser_registration
    EXPORT teaserpp-export
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
list(APPEND TEASERPP_EXPORTED_TARGETS teaser_registration pmc)
add_library(teaserpp::teaser_registration ALIAS teaser_registration)

# teaser_features library
if (BUILD_TEASER_FPFH)
    include_directories(${PCL_INCLUDE_DIRS})
    add_library(teaser_features SHARED
            src/fpfh.cc
            src/matcher.cc
            )
    target_link_libraries(teaser_features
            PRIVATE ${PCL_LIBRARIES}
            PRIVATE Eigen3::Eigen
            )
    if (BUILD_WITH_MKL AND MKL_FOUND)
        target_link_libraries(teaser_features
                PRIVATE ${MKL_LIBRARIES}
                )
    endif ()
    target_include_directories(teaser_features PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

    install(TARGETS teaser_features
        EXPORT teaserpp-export
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    list(APPEND TEASERPP_EXPORTED_TARGETS teaser_features)
    add_library(teaserpp::teaser_features ALIAS teaser_features)
endif ()

# march=native flag
if (BUILD_WITH_MARCH_NATIVE)
    message(STATUS "-march=native flag enabled.")
    target_compile_options(teaser_registration PUBLIC -march=native)
endif ()

# set exported targets in parent scope
set(TEASERPP_EXPORTED_TARGETS "${TEASERPP_EXPORTED_TARGETS}" PARENT_SCOPE)

# installation
install(EXPORT teaserpp-export 
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/teaserpp
    NAMESPACE teaserpp::
    FILE teaserppTargets.cmake
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${SPECTRA_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
