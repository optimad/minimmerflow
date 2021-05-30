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
cmake_minimum_required(VERSION 3.4)

#------------------------------------------------------------------------------------#
# Functions
#------------------------------------------------------------------------------------#

# Initialize test directory
function(initializeTestDirectory TEST_SETUP_TARGET TEST_DIRECTORY)
    getTestSetupTarget(TEST_SETUP_TARGET ${TEST_DIRECTORY})
    set(TEST_SETUP_TARGET ${TEST_SETUP_TARGET} PARENT_SCOPE)

    add_custom_target(${TEST_SETUP_TARGET} ALL)

    ADD_DEPENDENCIES(${TEST_SETUP_TARGET} ${MINIMMERFLOW_EXECUTABLE_NAME})

    # Copy the controldict file
    add_custom_command(
        TARGET ${TEST_SETUP_TARGET}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/settings.xml"
            "${CMAKE_CURRENT_BINARY_DIR}/settings.xml"
        MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/settings.xml"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/settings.xml"
    )
endfunction()

# Add a serial test
function(addSerialTest TEST_NAME TEST_EXPECTED_RESULTS WORKING_DIRECTORY)
    addTest("${TEST_NAME}" "${TEST_EXPECTED_RESULTS}" "${WORKING_DIRECTORY}" 1 1)
endfunction()

# Add a parallel MPI test
function(addParallelMPITest TEST_NAME TEST_EXPECTED_RESULTS WORKING_DIRECTORY)
    if (NOT MINIMMERFLOW_ENABLE_MPI)
        return()
    endif ()

    # Get the number of nProcesses to run the test on
    list(LENGTH ARGN ARGN_LENGTH)
    if (${ARGN_LENGTH} GREATER 0)
        list(GET ARGN 0 N_PROCS)
    else ()
        set(N_PROCS "${MPIEXEC_MAX_NUMPROCS}")
    endif ()

    # Add the test
    addTest("${TEST_NAME}" "${TEST_EXPECTED_RESULTS}" "${WORKING_DIRECTORY}" ${N_PROCS} 1)
endfunction()

# Add a parallel OpenMP test
function(addParallelOpenMPTest TEST_NAME TEST_EXPECTED_RESULTS WORKING_DIRECTORY)
    if (NOT MINIMMERFLOW_ENABLE_OPENMP)
        return()
    endif ()

    # Get the number of threads to run the test on
    list(LENGTH ARGN ARGN_LENGTH)
    if (${ARGN_LENGTH} GREATER 0)
        list(GET ARGN 0 N_THREADS)
    else ()
        set(N_THREADS "-1")
    endif ()

    # Add the test
    addTest("${TEST_NAME}" "${TEST_EXPECTED_RESULTS}" "${WORKING_DIRECTORY}" 1 ${N_THREADS})
endfunction()

# Add a test
function(addTest TEST_NAME TEST_EXPECTED_RESULTS WORKING_DIRECTORY N_PROCS N_THREADS)
    if (NOT MINIMMERFLOW_ENABLE_MPI AND ${N_PROCS} GREATER 1)
        return()
    endif ()

    if (NOT MINIMMERFLOW_ENABLE_OPENMP AND ${N_THREADS} GREATER 1)
        return()
    endif ()

    # Test command
    set(TEST_COMMAND "$<TARGET_FILE:${MINIMMERFLOW_EXECUTABLE_NAME}>")
    if (MINIMMERFLOW_ENABLE_MPI)
        if (MINIMMERFLOW_ENABLE_OPENMP)
            SET(MPIEXEC_ENVFLAGS "-x OMP_NUM_THREADS=1")
        endif ()

        set(TEST_COMMAND "${MPIEXEC} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${N_PROCS} ${MPIEXEC_POSTFLAGS} ${MPIEXEC_ENVFLAGS} ${TEST_COMMAND}")
    endif ()

    # Add test
    add_test(NAME ${TEST_NAME} COMMAND python3 ${CMAKE_BINARY_DIR}/test/test_driver.py --command "${TEST_COMMAND}" --expected "${TEST_EXPECTED_RESULTS}" WORKING_DIRECTORY "${WORKING_DIRECTORY}")

    # Set test variables
    if (MINIMMERFLOW_ENABLE_OPENMP)
        if (NOT ${N_THREADS} EQUAL -1)
            set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${N_THREADS}")
        endif ()
    endif ()

    if (${N_PROCS} GREATER 1)
        set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT OMPI_MCA_rmaps_base_oversubscribe=1)
    endif()
endfunction()

# Get the test setup target for the test
function(getTestSetupTarget TEST_SETUP_TARGET TEST_DIRECTORY)
    set(${TEST_SETUP_TARGET} "test_${TEST_DIRECTORY}_setup" PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------------#
# Subdirectories
#------------------------------------------------------------------------------------#

set(TEST_DIRECTORIES "")
list(APPEND TEST_DIRECTORIES "2dIsentropicVortex")
list(APPEND TEST_DIRECTORIES "2dRadSod")
list(APPEND TEST_DIRECTORIES "3dIsentropicVortexXY")
list(APPEND TEST_DIRECTORIES "3dIsentropicVortexYZ")
list(APPEND TEST_DIRECTORIES "3dRadSod")

add_custom_target("test_setup")
foreach (TEST_DIRECTORY IN LISTS TEST_DIRECTORIES)
    # Add directory
    add_subdirectory("${TEST_DIRECTORY}")

    # Update the dependencies of the test setup target
    getTestSetupTarget(TEST_SETUP_TARGET ${TEST_DIRECTORY})
    ADD_DEPENDENCIES(test_setup ${TEST_SETUP_TARGET})
endforeach()

#------------------------------------------------------------------------------------#
# Targets
#------------------------------------------------------------------------------------#

add_custom_target(check DEPENDS ${MINIMMERFLOW_EXECUTABLE_NAME} test_setup COMMAND ${CMAKE_MAKE_PROGRAM} test)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/test_driver.py"
    "${CMAKE_CURRENT_BINARY_DIR}/test_driver.py"
)
