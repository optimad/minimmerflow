/*---------------------------------------------------------------------------*\
 *
 *  minimmerflow
 *
 *  Copyright (C) 2015-2021 OPTIMAD engineering Srl
 *
 *  -------------------------------------------------------------------------
 *  License
 *  This file is part of minimmerflow.
 *
 *  minimmerflow is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License v3 (LGPL)
 *  as published by the Free Software Foundation.
 *
 *  minimmerflow is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with minimmerflow. If not, see <http://www.gnu.org/licenses/>.
 *
\*---------------------------------------------------------------------------*/

#ifndef __MINIMMERFLOW_PROBLEM_HPP__
#define __MINIMMERFLOW_PROBLEM_HPP__

#include "constants.hpp"
#include "mesh_info.hpp"

namespace problem {

enum ProblemType {
    PROBLEM_VORTEX_XY = 0,
    PROBLEM_VORTEX_ZX = 1,
    PROBLEM_VORTEX_YZ = 2,
    PROBLEM_RADSOD    = 3,
    PROBLEM_SOD_X     = 4,
    PROBLEM_SOD_Y     = 5,
    PROBLEM_SOD_Z     = 6,
    PROBLEM_FFSTEP    = 7
};

ProblemType getProblemType();

void getDomainData(ProblemType problemType, int &dimensions, std::array<double, 3> *origin, double *length);

double getStartTime(ProblemType problemType, int dimensions);
double getEndTime(ProblemType problemType, int dimensions);

void evalCellInitalConservatives(ProblemType problemType, long cellId, const MeshGeometricalInfo &meshInfo, double *primitives);
void evalCellExactConservatives(ProblemType problemType, long cellId, const MeshGeometricalInfo &meshInfo, double t, double *primitives);

void evalExactConservatives(ProblemType problemType, int dimensions, std::array<double, 3> point, double t, double *primitives);

int getBorderBCType(ProblemType problemType, long id, const MeshGeometricalInfo &meshInfo);

void getBorderBCInfo(ProblemType problemType, int BCType, const std::array<double, 3> &point,
                     const std::array<double, 3> &normal, std::array<double, BC_INFO_SIZE> &info);

}

#endif
