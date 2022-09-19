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
#include "reconstruction.hpp"
#include "storage.hpp"

#include <bitpit_voloctree.hpp>

namespace euler {

void evalSplitting(const double *conservativeL, const double *conservativeR, const double *n, double *fluxes, double *lambda);

void evalFluxes(const double *conservative, const double *primitive, const double *n, double *fluxes);

void computeRHS(problem::ProblemType problemType, const ComputationInfo &computationInfo,
                const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
                const ReconstructionCalculator &reconstructionCalculator,
                ScalarPiercedStorageCollection<double> *cellsRHS, double *maxEig);

#if ENABLE_CUDA
void cuda_initialize();
void cuda_finalize();
#endif

void resetRHS(ScalarPiercedStorageCollection<double> *cellsRHS);
#if ENABLE_CUDA
void cuda_resetRHS(ScalarPiercedStorageCollection<double> *cellsRHS);
#endif

void updateRHS(problem::ProblemType problemType,const ComputationInfo &computationInfo,
               const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
               const ReconstructionCalculator &reconstructionCalculator,
               ScalarPiercedStorageCollection<double> *cellsRHS, double *maxEig);
#if ENABLE_CUDA
void cuda_updateRHS(problem::ProblemType problemType, const ComputationInfo &computationInfo,
                    const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
                    const ReconstructionCalculator &reconstructionCalculator,
                    ScalarPiercedStorageCollection<double> *cellsRHS, double *maxEig);
#endif

void evalInterfaceBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                           problem::ProblemType problemType, int BCType, const double *innerValues,
                           double *boundaryValues);

void evalFreeFlowBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                          const std::array<double, BC_INFO_SIZE> &info, const double *innerValues,
                          double *boundaryValues);

void evalReflectingBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                            const std::array<double, BC_INFO_SIZE> &info, const double *innerValues,
                            double *boundaryValues);

void evalWallBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                      const std::array<double, BC_INFO_SIZE> &info, const double *innerValues,
                      double *boundaryValues);

void evalDirichletBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                           const std::array<double, BC_INFO_SIZE> &info, const double *innerValues,
                           double *boundaryValues);

}

#endif
