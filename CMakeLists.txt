cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project information
project(CahnHillardExample CXX)

# Append directory containing important modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

# libmesh
find_package(LibMesh REQUIRED)

# mpi
find_package(MPI REQUIRED)

# threads
find_package(Threads REQUIRED)

# vtk
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# debug output flags
message("LIBMESH_DEFINITIONS = ${PC_LIBMESH_CFLAGS_DEFS}")
message("LIBMESH_FLAGS = ${PC_LIBMESH_CFLAGS_FLAGS}")
message("LIBMESH_INCLUDE_DIRS = ${LIBMESH_INCLUDE_DIRS}")
message("LIBMESH_LIBRARIES = ${LIBMESH_LIBRARIES}")

add_subdirectory(src)
add_subdirectory(apps)

# unittests
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
message(STATUS "Enable testing: ${ENABLE_UNIT_TESTS}")
if (ENABLE_UNIT_TESTS)
    enable_testing()

    add_subdirectory(unittests)

    find_package(Catch2 QUIET)
    # if cmake cannot find the package, we download it
    if (NOT Catch2_FOUND)
        # get catch2 for testing
        include(FetchContent)
        FetchContent_Declare(
                catch2_content
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_TAG v2.13.1
        )
        FetchContent_MakeAvailable(catch2_content)
    endif()
endif()