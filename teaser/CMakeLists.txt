include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(pmc
    GIT_REPOSITORY https://github.com/jingnanshi/pmc.git
)
if (NOT pmc_POPULATED)
  FetchContent_Populate(pmc)
  set(PMC_BUILD_SHARED OFF CACHE INTERNAL "")
  add_subdirectory(${pmc_SOURCE_DIR} ${pmc_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(tinyply
    GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
if (NOT tinyply_POPULATED)
  FetchContent_Populate(tinyply)
endif()

FetchContent_Declare(spectra
    GIT_REPOSITORY    https://github.com/jingnanshi/spectra
    GIT_TAG           5c4fb1de050847988faaaaa50f60e7d3d5f16143
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
if (NOT spectra_POPULATED)
  FetchContent_Populate(spectra)
endif()

# teaser_io library
add_library(teaser_io src/ply_io.cc)
target_include_directories(teaser_io
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        $<BUILD_INTERFACE:${tinyply_SOURCE_DIR}/source>
)
if(NOT BUILD_SHARED_LIBS)
  set_property(TARGET teaser_io PROPERTY POSITION_INDEPENDENT_CODE 1)
endif()

add_library(teaserpp::teaser_io ALIAS teaser_io)

# teaser_registration library
add_library(teaser_registration
        src/certification.cc
        src/registration.cc
        src/graph.cc
)

target_link_libraries(teaser_registration
        PUBLIC Eigen3::Eigen
        PRIVATE pmc ${TEASERPP_BLAS_LAPACK_LIBS}
)
target_include_directories(teaser_registration
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        $<BUILD_INTERFACE:${spectra_SOURCE_DIR}/include>
)
if(NOT BUILD_SHARED_LIBS)
  set_property(TARGET teaser_registration PROPERTY POSITION_INDEPENDENT_CODE 1)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(teaser_registration PUBLIC OpenMP::OpenMP_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

add_library(teaserpp::teaser_registration ALIAS teaser_registration)

# teaser_features library
if (BUILD_TEASER_FPFH)
    include_directories(${PCL_INCLUDE_DIRS})
    add_library(teaser_features SHARED
            src/fpfh.cc
            src/matcher.cc
    )
    target_link_libraries(teaser_features
            PRIVATE ${PCL_LIBRARIES} Eigen3::Eigen
    )
    if (BUILD_WITH_MKL AND MKL_FOUND)
        target_link_libraries(teaser_features PRIVATE ${MKL_LIBRARIES})
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

# installation
if (NOT DEFINED SKBUILD)
  install(TARGETS teaser_io teaser_registration
          EXPORT teaserpp-export
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  list(APPEND TEASERPP_EXPORTED_TARGETS teaser_io teaser_registration)
  set(TEASERPP_EXPORTED_TARGETS "${TEASERPP_EXPORTED_TARGETS}" PARENT_SCOPE)

  install(EXPORT teaserpp-export
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/teaserpp
      NAMESPACE teaserpp::
      FILE teaserppTargets.cmake
  )

  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif ()
