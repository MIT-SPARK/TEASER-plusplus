# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:

FindLapack
----------

* Michael Hirsch, Ph.D. www.scivision.dev
* David Eklund

Let Michael know if there are more MKL / Lapack / compiler combination you want.
Refer to https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor

Finds LAPACK libraries for C / C++ / Fortran.
Works with Netlib Lapack / LapackE, Atlas and Intel MKL.
Intel MKL relies on having environment variable MKLROOT set, typically by sourcing
mklvars.sh beforehand.

Why not the FindLapack.cmake built into CMake? It has a lot of old code for
infrequently used Lapack libraries and is unreliable for me.

Tested on Linux, MacOS and Windows with:
* GCC / Gfortran
* Clang / Flang
* PGI (pgcc, pgfortran)
* Intel (icc, ifort)


Parameters
^^^^^^^^^^

COMPONENTS default to Netlib LAPACK / LapackE, otherwise:

``MKL``
  Intel MKL for MSVC, ICL, ICC, GCC and PGCC -- sequential by default, or add TBB or MPI as well
``OpenMP``
  Intel MPI with OpenMP threading addition to MKL
``TBB``
  Intel MPI + TBB for MKL
``MKL64``
  MKL only: 64-bit integers  (default is 32-bit integers)

``LAPACKE``
  Netlib LapackE for C / C++
``Netlib``
  Netlib Lapack for Fortran
``OpenBLAS``
  OpenBLAS Lapack for Fortran

``LAPACK95``
  get Lapack95 interfaces for MKL or Netlib (must also specify one of MKL, Netlib)


Result Variables
^^^^^^^^^^^^^^^^

``LAPACK_FOUND``
  Lapack libraries were found
``LAPACK_<component>_FOUND``
  LAPACK <component> specified was found
``LAPACK_LIBRARIES``
  Lapack library files (including BLAS
``LAPACK_INCLUDE_DIRS``
  Lapack include directories (for C/C++)


References
^^^^^^^^^^

* Pkg-Config and MKL:  https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-and-pkg-config-tool
* MKL for Windows: https://software.intel.com/en-us/mkl-windows-developer-guide-static-libraries-in-the-lib-intel64-win-directory
* MKL Windows directories: https://software.intel.com/en-us/mkl-windows-developer-guide-high-level-directory-structure
* Atlas http://math-atlas.sourceforge.net/errata.html#LINK
* MKL LAPACKE (C, C++): https://software.intel.com/en-us/mkl-linux-developer-guide-calling-lapack-blas-and-cblas-routines-from-c-c-language-environments
#]=======================================================================]

# clear to avoid endless appending on subsequent calls
set(LAPACK_LIBRARY)
set(LAPACK_INCLUDE_DIR)

# ===== functions ==========

function(atlas_libs)

find_library(ATLAS_LIB
  NAMES atlas
  NAMES_PER_DIR
  PATH_SUFFIXES atlas)

pkg_check_modules(pc_atlas_lapack lapack-atlas QUIET)

find_library(LAPACK_ATLAS
  NAMES ptlapack lapack_atlas lapack
  NAMES_PER_DIR
  PATH_SUFFIXES atlas
  HINTS ${pc_atlas_lapack_LIBRARY_DIRS} ${pc_atlas_lapack_LIBDIR})

pkg_check_modules(pc_atlas_blas blas-atlas QUIET)

find_library(BLAS_LIBRARY
  NAMES ptf77blas f77blas blas
  NAMES_PER_DIR
  PATH_SUFFIXES atlas
  HINTS ${pc_atlas_blas_LIBRARY_DIRS} ${pc_atlas_blas_LIBDIR})
# === C ===
find_library(BLAS_C_ATLAS
  NAMES ptcblas cblas
  NAMES_PER_DIR
  PATH_SUFFIXES atlas
  HINTS ${pc_atlas_blas_LIBRARY_DIRS} ${pc_atlas_blas_LIBDIR})

find_path(LAPACK_INCLUDE_DIR
  NAMES cblas-atlas.h cblas.h clapack.h
  HINTS ${pc_atlas_blas_INCLUDE_DIRS} ${pc_atlas_blas_LIBDIR})

#===========
if(LAPACK_ATLAS AND BLAS_C_ATLAS AND BLAS_LIBRARY AND ATLAS_LIB)
  set(LAPACK_Atlas_FOUND true PARENT_SCOPE)
  set(LAPACK_LIBRARY ${LAPACK_ATLAS} ${BLAS_C_ATLAS} ${BLAS_LIBRARY} ${ATLAS_LIB})
  if(NOT WIN32)
    find_package(Threads)  # not required--for example Flang
    list(APPEND LAPACK_LIBRARY ${CMAKE_THREAD_LIBS_INIT})
  endif()
endif()

set(LAPACK_LIBRARY ${LAPACK_LIBRARY} PARENT_SCOPE)
set(LAPACK_INCLUDE_DIR ${LAPACK_INCLUDE_DIR} PARENT_SCOPE)

endfunction(atlas_libs)

#=======================

function(netlib_libs)

if(LAPACK95 IN_LIST LAPACK_FIND_COMPONENTS)
  find_path(LAPACK95_INCLUDE_DIR
              NAMES f95_lapack.mod
              PATH_SUFFIXES include
              PATHS ${LAPACK95_ROOT})

  find_library(LAPACK95_LIBRARY
                 NAMES lapack95
                 NAMES_PER_DIR
                 PATH_SUFFIXES lib
                 PATHS ${LAPACK95_ROOT})

  if(LAPACK95_LIBRARY AND LAPACK95_INCLUDE_DIR)
    set(LAPACK_INCLUDE_DIR ${LAPACK95_INCLUDE_DIR})
    set(LAPACK_LIBRARY ${LAPACK95_LIBRARY})
  else()
    return()
  endif()
endif(LAPACK95 IN_LIST LAPACK_FIND_COMPONENTS)

set(_lapack_hints)
if(CMAKE_Fortran_COMPILER_ID STREQUAL PGI)
  get_filename_component(_pgi_path ${CMAKE_Fortran_COMPILER} DIRECTORY)
  set(_lapack_hints ${_pgi_path}/../)
endif()

pkg_check_modules(pc_lapack lapack-netlib QUIET)
if(NOT pc_lapack_FOUND)
  pkg_check_modules(pc_lapack lapack QUIET)  # Netlib on Cygwin, Homebrew and others
endif()
find_library(LAPACK_LIB
  NAMES lapack
  NAMES_PER_DIR
  PATHS /usr/local/opt  # homebrew
  HINTS ${_lapack_hints} ${pc_lapack_LIBRARY_DIRS} ${pc_lapack_LIBDIR}
  PATH_SUFFIXES lib lapack lapack/lib)
if(LAPACK_LIB)
  list(APPEND LAPACK_LIBRARY ${LAPACK_LIB})
else()
  return()
endif()

if(LAPACKE IN_LIST LAPACK_FIND_COMPONENTS)
  pkg_check_modules(pc_lapacke lapacke QUIET)
  find_library(LAPACKE_LIBRARY
    NAMES lapacke
    NAMES_PER_DIR
    PATHS /usr/local/opt
    HINTS ${pc_lapacke_LIBRARY_DIRS} ${pc_lapacke_LIBDIR}
    PATH_SUFFIXES lapack lapack/lib)

  # lapack/include for Homebrew
  find_path(LAPACKE_INCLUDE_DIR
    NAMES lapacke.h
    PATHS /usr/local/opt
    HINTS ${pc_lapacke_INCLUDE_DIRS} ${pc_lapacke_LIBDIR}
    PATH_SUFFIXES lapack lapack/include)

  if(LAPACKE_LIBRARY AND LAPACKE_INCLUDE_DIR)
    set(LAPACK_LAPACKE_FOUND true PARENT_SCOPE)
    list(APPEND LAPACK_INCLUDE_DIR ${LAPACKE_INCLUDE_DIR})
    list(APPEND LAPACK_LIBRARY ${LAPACKE_LIBRARY})
  else()
    message(WARNING "Trouble finding LAPACKE:
      include: ${LAPACKE_INCLUDE_DIR}
      libs: ${LAPACKE_LIBRARY}")
    return()
  endif()

  mark_as_advanced(LAPACKE_LIBRARY LAPACKE_INCLUDE_DIR)
endif(LAPACKE IN_LIST LAPACK_FIND_COMPONENTS)

pkg_check_modules(pc_blas blas-netlib QUIET)
if(NOT pc_blas_FOUND)
  pkg_check_modules(pc_blas blas QUIET)  # Netlib on Cygwin and others
endif()
find_library(BLAS_LIBRARY
  NAMES refblas blas
  NAMES_PER_DIR
  PATHS /usr/local/opt
  HINTS ${_lapack_hints} ${pc_blas_LIBRARY_DIRS} ${pc_blas_LIBDIR}
  PATH_SUFFIXES lib lapack lapack/lib blas)

if(BLAS_LIBRARY)
  list(APPEND LAPACK_LIBRARY ${BLAS_LIBRARY})
  set(LAPACK_Netlib_FOUND true PARENT_SCOPE)
else()
  return()
endif()

if(NOT WIN32)
  list(APPEND LAPACK_LIBRARY ${CMAKE_THREAD_LIBS_INIT})
endif()

if(LAPACK95_LIBRARY)
  set(LAPACK_LAPACK95_FOUND true PARENT_SCOPE)
endif()

set(LAPACK_LIBRARY ${LAPACK_LIBRARY} PARENT_SCOPE)
set(LAPACK_INCLUDE_DIR ${LAPACK_INCLUDE_DIR} PARENT_SCOPE)

endfunction(netlib_libs)

#===============================
function(openblas_libs)

pkg_check_modules(pc_lapack lapack-openblas QUIET)
find_library(LAPACK_LIBRARY
  NAMES lapack
  NAMES_PER_DIR
  HINTS ${pc_lapack_LIBRARY_DIRS} ${pc_lapack_LIBDIR}
  PATH_SUFFIXES openblas)


pkg_check_modules(pc_blas blas-openblas QUIET)
find_library(BLAS_LIBRARY
  NAMES openblas blas
  NAMES_PER_DIR
  HINTS ${pc_blas_LIBRARY_DIRS} ${pc_blas_LIBDIR}
  PATH_SUFFIXES openblas)

find_path(LAPACK_INCLUDE_DIR
    NAMES cblas-openblas.h cblas.h f77blas.h openblas_config.h
    HINTS ${pc_lapack_INCLUDE_DIRS})

if(LAPACK_LIBRARY AND BLAS_LIBRARY)
  list(APPEND LAPACK_LIBRARY ${BLAS_LIBRARY})
  set(LAPACK_OpenBLAS_FOUND true PARENT_SCOPE)
else()
  message(WARNING "Trouble finding OpenBLAS:
      include: ${LAPACK_INCLUDE_DIR}
      libs: ${LAPACK_LIBRARY} ${BLAS_LIBRARY}")
  return()
endif()

if(NOT WIN32)
 find_package(Threads)  # not required--for example Flang
 list(APPEND LAPACK_LIBRARY ${CMAKE_THREAD_LIBS_INIT})
endif()

set(LAPACK_LIBRARY ${LAPACK_LIBRARY} PARENT_SCOPE)
set(LAPACK_INCLUDE_DIR ${LAPACK_INCLUDE_DIR} PARENT_SCOPE)

endfunction(openblas_libs)

#===============================

function(find_mkl_libs)
# https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor

set(_mkl_libs ${ARGV})
if((UNIX AND NOT APPLE) AND CMAKE_Fortran_COMPILER_ID STREQUAL GNU)
  list(INSERT _mkl_libs 0 mkl_gf_${_mkl_bitflag}lp64)
else()
  list(INSERT _mkl_libs 0 mkl_intel_${_mkl_bitflag}lp64)
endif()

# Note: Don't remove items from PATH_SUFFIXES unless you're extensively testing,
# each path is there for a specific reason!

foreach(s ${_mkl_libs})
  find_library(LAPACK_${s}_LIBRARY
           NAMES ${s}
           NAMES_PER_DIR
           PATHS ${MKLROOT} ENV TBBROOT
           PATH_SUFFIXES
             lib lib/intel64 lib/intel64_win
             lib/intel64/gcc4.7 ../tbb/lib/intel64/gcc4.7
             lib/intel64/vc_mt ../tbb/lib/intel64/vc_mt
             ../compiler/lib/intel64
           HINTS ${pc_mkl_LIBRARY_DIRS} ${pc_mkl_LIBDIR}
           NO_DEFAULT_PATH)

  if(NOT LAPACK_${s}_LIBRARY)
    message(STATUS "MKL component not found: " ${s})
    return()
  endif()

  list(APPEND LAPACK_LIB ${LAPACK_${s}_LIBRARY})
endforeach()

if(NOT BUILD_SHARED_LIBS AND (UNIX AND NOT APPLE))
  set(LAPACK_LIB -Wl,--start-group ${LAPACK_LIB} -Wl,--end-group)
endif()

set(BLAS_LIBRARY PARENT_SCOPE)
set(LAPACK_LIBRARY ${LAPACK_LIB} PARENT_SCOPE)
set(LAPACK_INCLUDE_DIR
  ${MKLROOT}/include
  ${MKLROOT}/include/intel64/${_mkl_bitflag}lp64
  PARENT_SCOPE)
 # ${pc_mkl_INCLUDE_DIRS} has garbage on Windows

endfunction(find_mkl_libs)

# ========== main program

if(NOT (OpenBLAS IN_LIST LAPACK_FIND_COMPONENTS
  OR Netlib IN_LIST LAPACK_FIND_COMPONENTS
  OR Atlas IN_LIST LAPACK_FIND_COMPONENTS
  OR MKL IN_LIST LAPACK_FIND_COMPONENTS))
  if(DEFINED ENV{MKLROOT})
    list(APPEND LAPACK_FIND_COMPONENTS MKL)
  else()
    list(APPEND LAPACK_FIND_COMPONENTS Netlib)
  endif()
endif()

find_package(PkgConfig QUIET)

# ==== generic MKL variables ====

if(MKL IN_LIST LAPACK_FIND_COMPONENTS)
  # we have to sanitize MKLROOT if it has Windows backslashes (\) otherwise it will break at build time
  # double-quotes are necessary per CMake to_cmake_path docs.
  file(TO_CMAKE_PATH "$ENV{MKLROOT}" MKLROOT)

  list(APPEND CMAKE_PREFIX_PATH ${MKLROOT}/bin/pkgconfig)

  if(NOT WIN32)
    find_package(Threads)
  endif()

  if(BUILD_SHARED_LIBS)
    set(_mkltype dynamic)
  else()
    set(_mkltype static)
  endif()

  if(MKL64 IN_LIST LAPACK_FIND_COMPONENTS)
    set(_mkl_bitflag i)
  else()
    set(_mkl_bitflag)
  endif()

  unset(_mkl_libs)
  if(LAPACK95 IN_LIST LAPACK_FIND_COMPONENTS)
    list(APPEND _mkl_libs mkl_blas95_${_mkl_bitflag}lp64 mkl_lapack95_${_mkl_bitflag}lp64)
  endif()

  unset(_tbb)
  if(TBB IN_LIST LAPACK_FIND_COMPONENTS)
    list(APPEND _mkl_libs mkl_tbb_thread mkl_core)
    set(_tbb tbb stdc++)
    if(WIN32)
      list(APPEND _mkl_libs tbb.lib)
    endif()
  elseif(OpenMP IN_LIST LAPACK_FIND_COMPONENTS)
    pkg_check_modules(pc_mkl mkl-${_mkltype}-${_mkl_bitflag}lp64-iomp QUIET)

    set(_mp iomp5)
    if(WIN32)
      set(_mp libiomp5md)  # "lib" is indeed necessary, even on CMake 3.14.0
    endif()
    list(APPEND _mkl_libs mkl_intel_thread mkl_core ${_mp})
  else()
    pkg_check_modules(pc_mkl mkl-${_mkltype}-${_mkl_bitflag}lp64-seq QUIET)
    list(APPEND _mkl_libs mkl_sequential mkl_core)
  endif()

  find_mkl_libs(${_mkl_libs})

  if(LAPACK_LIBRARY)

    if(NOT WIN32)
      list(APPEND LAPACK_LIBRARY ${_tbb} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} m)
    endif()

    set(LAPACK_MKL_FOUND true)

    if(MKL64 IN_LIST LAPACK_FIND_COMPONENTS)
      set(LAPACK_MKL64_FOUND true)
    endif()

    if(LAPACK95 IN_LIST LAPACK_FIND_COMPONENTS)
      set(LAPACK_LAPACK95_FOUND true)
    endif()

    if(OpenMP IN_LIST LAPACK_FIND_COMPONENTS)
      set(LAPACK_OpenMP_FOUND true)
    endif()

    if(TBB IN_LIST LAPACK_FIND_COMPONENTS)
      set(LAPACK_TBB_FOUND true)
    endif()
  endif()

elseif(Atlas IN_LIST LAPACK_FIND_COMPONENTS)

  atlas_libs()

elseif(Netlib IN_LIST LAPACK_FIND_COMPONENTS)

  netlib_libs()

elseif(OpenBLAS IN_LIST LAPACK_FIND_COMPONENTS)

  openblas_libs()

endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACK
  REQUIRED_VARS LAPACK_LIBRARY
  HANDLE_COMPONENTS)

set(BLAS_LIBRARIES ${BLAS_LIBRARY})
set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})
set(LAPACK_INCLUDE_DIRS ${LAPACK_INCLUDE_DIR})

if(LAPACK_FOUND)
# need if _FOUND guard to allow project to autobuild; can't overwrite imported target even if bad
  if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES
                          INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARY}"
                        )
  endif()

  if(NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)
    set_target_properties(LAPACK::LAPACK PROPERTIES
                          INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${LAPACK_INCLUDE_DIR}"
                        )
  endif()
endif()

mark_as_advanced(LAPACK_LIBRARY LAPACK_INCLUDE_DIR)
