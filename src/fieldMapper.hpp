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

#ifndef __MINIMMERFLOW_FIELDMAPPER_HPP__
#define __MINIMMERFLOW_FIELDMAPPER_HPP__

#include "computation_info.hpp"
#include "containers.hpp"
#include <bitpit_voloctree.hpp>

namespace fieldMapper {

#if ENABLE_CUDA
void cuda_mapField(std::size_t *previousIDs, std::size_t *currentIDs,
                   double *field);
#endif

void cpu_mapField(ScalarStorage<std::size_t> &previousIDs,
              ScalarStorage<std::size_t> &currentIDs,
              ScalarPiercedStorage<double> &field);

void mapFields(ScalarStorage<std::size_t> &previousIDs,
               ScalarStorage<std::size_t> &currentIDs,
               ScalarPiercedStorage<double> &cellRHS,
               ScalarPiercedStorage<double> &cellConservatives,
               ScalarPiercedStorage<double> &cellPrimitives,
               ScalarPiercedStorage<double> &cellConservativesWork,
               ComputationInfo &computationInfo);

}

#endif

