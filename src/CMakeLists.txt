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

# Name your project here
set(MINIMMERFLOW_EXECUTABLE_NAME minimmerflow CACHE INTERNAL "Executable name of the solver" FORCE)

file(GLOB sources "*.cpp")
add_executable(${MINIMMERFLOW_EXECUTABLE_NAME} ${sources})

target_link_libraries(${MINIMMERFLOW_EXECUTABLE_NAME} ${BITPIT_LIBRARIES})

INSTALL(TARGETS ${MINIMMERFLOW_EXECUTABLE_NAME} DESTINATION bin)
