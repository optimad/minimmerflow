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

# Initialize test directory
get_filename_component(TEST_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} NAME)
initializeTestDirectory(TEST_SETUP_TARGET, ${TEST_DIRECTORY})

# Reference error
set(3D_ISENTROPIC_VORTEX_YZ_RESULTS "1.182056784168e+01")

# Serial tests
addSerialTest("3dIsentropicVortexYZ" "${3D_ISENTROPIC_VORTEX_YZ_RESULTS}" "${CMAKE_CURRENT_BINARY_DIR}")

# Parallel - MPI
if (MINIMMERFLOW_ENABLE_MPI)
    addParallelMPITest("3dIsentropicVortexYZ_parallel_MPI" "${3D_ISENTROPIC_VORTEX_YZ_RESULTS}" "${CMAKE_CURRENT_BINARY_DIR}" 3)
endif ()
