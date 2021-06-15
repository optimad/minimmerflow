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

#ifndef __MINIMMERFLOW_EULER_HPP__
#define __MINIMMERFLOW_EULER_HPP__

#include "constants.hpp"
#include "mesh_info.hpp"
#include "problem.hpp"
#include "storage.hpp"

#include <bitpit_voloctree.hpp>

namespace euler {

//void evalSplitting(const double *conservativeL, const double *conservativeR, const std::array<double, 3> &n, FluxData *fluxes, double *lambda);
void evalSplitting
(
    const double *conservativeL,
    const double *conservativeR,
    const std::array<double, 3> &n,
    std::vector<double> *fluxesVec,
    double *lambda,
    const std::size_t interfaceRawId
);

void evalFluxes(const double *conservative, const double *primitive, const std::array<double, 3> &n, FluxData *fluxes);

void computeRHS(problem::ProblemType problemType, const MeshGeometricalInfo &meshInfo, const CellStorageBool &cellSolvedFlag,
                const int order, const CellStorageDouble &cellConservatives, const InterfaceStorageInt &interfaceBCs,
                CellStorageDouble *cellsRHS, double *maxEig);

void evalInterfaceBCValues(problem::ProblemType problemType, int BCType,
                           const std::array<double, 3> &point,
                           const std::array<double, 3> &normal,
                           const double *conservative, double *conservative_BC);

void evalFreeFlowBCValues(const std::array<double, 3> &point,
                          const std::array<double, 3> &normal,
                          const std::array<double, BC_INFO_SIZE> &info,
                          const double *conservative, double *conservative_BC);

void evalReflectingBCValues(const std::array<double, 3> &point,
                          const std::array<double, 3> &normal,
                          const std::array<double, BC_INFO_SIZE> &info,
                          const double *conservative, double *conservative_BC);

void evalWallBCValues(const std::array<double, 3> &point,
                          const std::array<double, 3> &normal,
                          const std::array<double, BC_INFO_SIZE> &info,
                          const double *conservative, double *conservative_BC);

void evalDirichletBCValues(const std::array<double, 3> &point,
                          const std::array<double, 3> &normal,
                          const std::array<double, BC_INFO_SIZE> &info,
                          const double *conservative, double *conservative_BC);

}

#endif
