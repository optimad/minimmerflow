#---------------------------------------------------------------------------
#
#  minimmerflow
#
#  Copyright (C) 2015-2021 OPTIMAD engineering Srl
#
#  -------------------------------------------------------------------------
#  License
#  This file is part of minimmerflow.
#
#  minimmerflow is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License v3 (LGPL)
#  as published by the Free Software Foundation.
#
#  minimmerflow is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
#  License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with minimmerflow. If not, see <http://www.gnu.org/licenses/>.
#
#---------------------------------------------------------------------------*/

# Specify the version being used as well as the language
cmake_minimum_required(VERSION 3.13)

# Name your project here
project(minimmerflow)

#------------------------------------------------------------------------------------#
# Variables visible to the user
#------------------------------------------------------------------------------------#
set(MINIMMERFLOW_ENABLE_MPI 0 CACHE BOOL "If set, the program is compiled with MPI support")
set(MINIMMERFLOW_ENABLE_OPENACC 0 CACHE BOOL "If set, the program is compiled with OpenACC support")
set(MINIMMERFLOW_ENABLE_CUDA 0 CACHE BOOL "If set, the program is compiled with CUDA support")
set(MINIMMERFLOW_VERBOSE_MAKE 0 CACHE BOOL "Set appropriate compiler and cmake flags to enable verbose output from compilation")

if (MINIMMERFLOW_ENABLE_OPENACC)
    set(OpenACC_ACCEL_TARGET "" CACHE STRING "OpenACC target accelerator")
endif()

if (MINIMMERFLOW_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "OFF" CACHE STRING "List of CUDA architectures (e.g., \"50 60 72\"). If set to OFF, it will not pass architectures.")
    endif()
endif()

#------------------------------------------------------------------------------------#
# External utilities
#------------------------------------------------------------------------------------#
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

#------------------------------------------------------------------------------------#
# Customized build types
#------------------------------------------------------------------------------------#

# Set build type variable
if(NOT CMAKE_BUILD_TYPE)
    # Set default build type to Debug
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo RelWithSanitize MinSizeRel."
        FORCE)

    # Set the possible values of build type for the GUI
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "RelWithSanitize")
endif()

#------------------------------------------------------------------------------------#
# Internal dependencies
#------------------------------------------------------------------------------------#
include_directories("src")

#------------------------------------------------------------------------------------#
# Enable needed languages
#------------------------------------------------------------------------------------#
if (MINIMMERFLOW_ENABLE_CUDA)
    enable_language(CUDA)
endif()

#------------------------------------------------------------------------------------#
# External dependencies
#------------------------------------------------------------------------------------#
if (MINIMMERFLOW_ENABLE_MPI)
    find_package(MPI)
endif()

if (MINIMMERFLOW_ENABLE_OPENACC)
    find_package(OpenACC REQUIRED)

    foreach(lang IN LISTS LANGUAGES)
        if (CMAKE_VERSION VERSION_LESS "3.16")
            string(REPLACE " " "\;" default_options ${OpenACC_${lang}_FLAGS})
            set(OpenACC_${lang}_OPTIONS ${default_options})
        endif()

        message(STATUS "OpenACC ${lang} version " ${OpenACC_${lang}_VERSION})
        message(STATUS "OpenACC ${lang} flags " ${OpenACC_${lang}_FLAGS})
    endforeach()
endif()

find_package(BITPIT REQUIRED)
include(${BITPIT_USE_FILE})

#------------------------------------------------------------------------------------#
# Compiler settings
#------------------------------------------------------------------------------------#
set (MINIMMERFLOW_DEFINITIONS "")

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)

set (ENABLE_WARNINGS ${MINIMMERFLOW_VERBOSE_MAKE})
if (MINIMMERFLOW_VERBOSE_MAKE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

if (MINIMMERFLOW_ENABLE_OPENACC)
    list (APPEND MINIMMERFLOW_DEFINITIONS "ENABLE_OPENACC=1")
endif()

if (MINIMMERFLOW_ENABLE_CUDA)
    list (APPEND MINIMMERFLOW_DEFINITIONS "ENABLE_CUDA=1")

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    # Return useful info from CUDA kernel build
    if (MINIMMERFLOW_VERBOSE_MAKE)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
    endif()

    # Add debug flags
    #
    # With these flags NVIDIA Nsight Compute is able to automaticall import
    # source code when run with the follwoing command:
    #   mpirun -np 1 ncu --set full --import-source=on -f -o output <executable>
    if (CMAKE_BUILD_TYPE_LOWER MATCHES "debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    endif()
endif()

if (MINIMMERFLOW_ENABLE_MPI)
    list (APPEND MINIMMERFLOW_DEFINITIONS "ENABLE_MPI=1")

    if (MPI_C_FOUND)
        set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
    else()
        set(CMAKE_C_COMPILER mpicc)
    endif()

    if (MPI_CXX_FOUND)
        set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    else()
        set(CMAKE_CXX_COMPILER mpic++)
    endif()
else ()
    list (APPEND MINIMMERFLOW_DEFINITIONS "ENABLE_MPI=0")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

if (MINIMMERFLOW_ENABLE_OPENACC)
    set(OpenACC_C_FLAGS_CUSTOM "")
    check_c_compiler_flag(-acc=noautopar OpenACC_C_SUPPORTS_NOAUTOPAR)
    if (OpenACC_C_SUPPORTS_NOAUTOPAR)
        set(OpenACC_C_FLAGS_CUSTOM "${OpenACC_C_FLAGS_CUSTOM} -acc=noautopar")
    endif()
    if (ENABLE_WARNINGS)
        check_c_compiler_flag(-Minfo=accel OpenACC_C_SUPPORTS_MINFO)
        if (OpenACC_C_SUPPORTS_MINFO)
            set(OpenACC_C_FLAGS_CUSTOM "${OpenACC_C_FLAGS_CUSTOM} -Minfo=accel")
        endif()
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenACC_C_FLAGS} ${OpenACC_C_FLAGS_CUSTOM}")

    set(OpenACC_CXX_FLAGS_CUSTOM "")
    check_cxx_compiler_flag(-acc=noautopar OpenACC_CXX_SUPPORTS_NOAUTOPAR)
    if (OpenACC_CXX_SUPPORTS_NOAUTOPAR)
        set(OpenACC_CXX_FLAGS_CUSTOM "${OpenACC_CXX_FLAGS_CUSTOM} -acc=noautopar")
    endif()
    if (ENABLE_WARNINGS)
        check_cxx_compiler_flag(-Minfo=accel OpenACC_CXX_SUPPORTS_MINFO)
        if (OpenACC_CXX_SUPPORTS_MINFO)
            set(OpenACC_CXX_FLAGS_CUSTOM "${OpenACC_CXX_FLAGS_CUSTOM} -Minfo=accel")
        endif()
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS} ${OpenACC_CXX_FLAGS_CUSTOM}")
endif()

if (ENABLE_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

if (NOT ("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}" VERSION_LESS "2.8.12"))
    add_compile_options("-std=c++11")
else ()
    add_definitions("-std=c++11")
endif ()

# Define a preprocessor macro to recognize debug builds
IF(CMAKE_BUILD_TYPE_LOWER MATCHES "debug" OR CMAKE_BUILD_TYPE_LOWER MATCHES "debinfo")
    list (APPEND MINIMMERFLOW_DEFINITIONS "ENABLE_DEBUG=1")
else ()
    list (APPEND MINIMMERFLOW_DEFINITIONS "ENABLE_DEBUG=0")
endif ()

IF(NOT CMAKE_BUILD_TYPE_LOWER MATCHES "debug")
    list (APPEND MINIMMERFLOW_DEFINITIONS "NDEBUG")
endif ()

# Add the definitions
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS ${MINIMMERFLOW_DEFINITIONS})

#------------------------------------------------------------------------------------#
# Version
#------------------------------------------------------------------------------------#
set (VERSION_HEADER_FILENAME "${PROJECT_SOURCE_DIR}/src/${CMAKE_PROJECT_NAME}_version.hpp")
string(TOUPPER "${CMAKE_PROJECT_NAME}_VERSION" VERSION_DEFINE_NAME)

file(READ "${VERSION_HEADER_FILENAME}" header)

string(REGEX REPLACE ".*#[ \t]*define[ \t]*${VERSION_DEFINE_NAME}[ \t]*\"([^\n]*)\".*" "\\1" match "${header}")
set(MINIMMERFLOW_VERSION "${match}")

string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\.[0-9]+(-[0-9A-Za-z-]+)?" "\\1" match "${MINIMMERFLOW_VERSION}")
set(MINIMMERFLOW_MAJOR_VERSION "${match}")

string(REGEX REPLACE "^[0-9]+\\.([0-9])+\\.[0-9]+(-[0-9A-Za-z-]+)?" "\\1" match "${MINIMMERFLOW_VERSION}")
set(MINIMMERFLOW_MINOR_VERSION "${match}")

string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+)(-[0-9A-Za-z-]+)?" "\\1" match "${MINIMMERFLOW_VERSION}")
set(MINIMMERFLOW_PATCH_VERSION "${match}")

string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+-([0-9A-Za-z-]+)" match "${MINIMMERFLOW_VERSION}")
if (NOT match STREQUAL "")
    string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.[0-9]+-([0-9A-Za-z-]+)" "\\1" match "${MINIMMERFLOW_VERSION}")
    set(MINIMMERFLOW_TAG_VERSION "${match}")
else ()
    set(MINIMMERFLOW_TAG_VERSION "")
endif ()

#------------------------------------------------------------------------------------#
# Subdirectories
#------------------------------------------------------------------------------------#

# Main program
add_subdirectory(src)

# Tests
enable_testing()
add_subdirectory(test)
