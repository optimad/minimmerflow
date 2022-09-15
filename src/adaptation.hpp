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

void markCellsForRefinement(bitpit::VolOctree &mesh, const double time, int &nCellsToBeRefined, const std::array<double,2> initialOrigin,
                            int &initialRefinementLevel, int &maxRefinementLevel);

void meshAdaptation(bitpit::VolOctree &mesh, std::vector<std::vector<double>> &parentCellRHS, std::vector<std::vector<double>> &parentCellConservatives,
                    ScalarPiercedStorageCollection<double> &cellRHS, ScalarPiercedStorageCollection<double> &cellConservatives, const double time,
                    const std::array<double,2> initialOrigin, int &initialRefinementLevel, int &maxRefinementLevel);

#if ENABLE_CUDA
void mapField(ScalarStorage<std::size_t> &parentIDs,
              ScalarStorage<std::size_t> &currentIDs,
              ScalarStorageCollection<double> &parentField,
              ScalarPiercedStorageCollection<double> &field);

void cuda_storeParentField(ScalarStorage<std::size_t> &parentIDs,
                           ScalarStorageCollection<double> &parentField,
                           ScalarPiercedStorageCollection<double> &field);

void cuda_mapField(ScalarStorage<std::size_t> &parentIDs,
                   ScalarStorage<std::size_t> &currentIDs,
                   ScalarStorageCollection<double> &parentField,
                   ScalarPiercedStorageCollection<double> &field);
#endif

void mapFields(ScalarStorage<std::size_t> &parentIDs,
               ScalarStorage<std::size_t> &currentIDs,
               ScalarStorageCollection<double> &parentCellRHS,
               ScalarStorageCollection<double> &parentCellConservatives,
               ScalarPiercedStorageCollection<double> &cellRHS,
               ScalarPiercedStorageCollection<double> &cellConservatives);

}

#endif
