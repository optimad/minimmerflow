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

#include "computation_info.hpp"
#include "constants.hpp"
#include "mesh_info.hpp"
#include "problem.hpp"
#include "storage.hpp"

#include <bitpit_voloctree.hpp>

namespace euler {

void evalSplitting(const float *conservativeL, const float *conservativeR, const float *n, float *fluxes, float *lambda);

void evalFluxes(const float *conservative, const float *primitive, const float *n, float *fluxes);

void computeRHS(problem::ProblemType problemType, ComputationInfo &computationInfo,
                const int order, const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
                const ScalarPiercedStorage<float> &cellConservatives, ScalarPiercedStorage<float> *cellsRHS, float *maxEig);

#if ENABLE_CUDA
void cuda_initialize();
void cuda_finalize();
#endif

void resetRHS(ScalarPiercedStorage<float> *cellsRHS);
#if ENABLE_CUDA
void cuda_resetRHS(ScalarPiercedStorage<float> *cellsRHS);
#endif

void updateRHS(problem::ProblemType problemType, ComputationInfo &computationInfo,
               const int order, const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
               const ScalarPiercedStorage<float> &cellConservatives, ScalarPiercedStorage<float> *cellsRHS, float *maxEig);
#if ENABLE_CUDA
void cuda_updateRHS(problem::ProblemType problemType, ComputationInfo &computationInfo,
                    const int order, const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
                    const ScalarPiercedStorage<float> &cellConservatives, ScalarPiercedStorage<float> *cellsRHS, float *maxEig);
#endif

void evalInterfaceBCValues(problem::ProblemType problemType, int BCType,
                           const std::array<float, 3> &point,
                           const std::array<float, 3> &normal,
                           const float *conservative, float *conservative_BC);

void evalFreeFlowBCValues(const std::array<float, 3> &point,
                          const std::array<float, 3> &normal,
                          const std::array<float, BC_INFO_SIZE> &info,
                          const float *conservative, float *conservative_BC);

void evalReflectingBCValues(const std::array<float, 3> &point,
                          const std::array<float, 3> &normal,
                          const std::array<float, BC_INFO_SIZE> &info,
                          const float *conservative, float *conservative_BC);

void evalWallBCValues(const std::array<float, 3> &point,
                          const std::array<float, 3> &normal,
                          const std::array<float, BC_INFO_SIZE> &info,
                          const float *conservative, float *conservative_BC);

void evalDirichletBCValues(const std::array<float, 3> &point,
                          const std::array<float, 3> &normal,
                          const std::array<float, BC_INFO_SIZE> &info,
                          const float *conservative, float *conservative_BC);

}

#endif
