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

#ifndef __MINIMMERFLOW_ADAPTATION_HPP__
#define __MINIMMERFLOW_ADAPTATION_HPP__

#include "containers.hpp"
#include "bitpit_common.hpp"
#include <bitpit_voloctree.hpp>

namespace adaptation {

void markCellsForRefinement(bitpit::VolOctree &mesh);

void meshAdaptation(bitpit::VolOctree &mesh,
                    ScalarStorage<std::size_t> &parentIDs,
                    ScalarStorage<std::size_t> &currentIDs,
                    ScalarStorage<double> &parentCellRHS,
                    ScalarStorage<double> &parentCellConservatives,
                    ScalarPiercedStorage<double> &cellRHS,
                    ScalarPiercedStorage<double> &cellConservatives);

#if ENABLE_CUDA
void mapField(ScalarStorage<std::size_t> &parentIDs,
              ScalarStorage<std::size_t> &currentIDs,
              ScalarStorage<double> &parentField,
              ScalarPiercedStorage<double> &field);

void cuda_storeParentField(ScalarStorage<std::size_t> &parentIDs,
                           ScalarStorage<double> &parentField,
                           ScalarPiercedStorage<double> &field);

void cuda_mapField(ScalarStorage<std::size_t> &parentIDs,
                   ScalarStorage<std::size_t> &currentIDs,
                   ScalarStorage<double> &parentField,
                   ScalarPiercedStorage<double> &field);
#endif

void mapFields(ScalarStorage<std::size_t> &parentIDs,
               ScalarStorage<std::size_t> &currentIDs,
               ScalarStorage<double> &parentCellRHS,
               ScalarStorage<double> &parentCellConservatives,
               ScalarPiercedStorage<double> &cellRHS,
               ScalarPiercedStorage<double> &cellConservatives);

}

#endif
