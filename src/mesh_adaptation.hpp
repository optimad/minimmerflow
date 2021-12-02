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

#ifndef __MINIMMERFLOW_MESHADAPTATION_HPP__
#define __MINIMMERFLOW_MESHADAPTATION_HPP__

#include "containers.hpp"
#include <bitpit_voloctree.hpp>

namespace mesh_adaptation {

void markCellsForRefinement(bitpit::VolOctree &mesh);

void meshAdaptation(bitpit::VolOctree &mesh,
                    ScalarStorage<std::size_t> &previousIDs,
                    ScalarStorage<std::size_t> &currentIDs);

}

#endif
