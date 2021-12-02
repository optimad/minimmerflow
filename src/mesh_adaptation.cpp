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

#include "mesh_adaptation.hpp"

using namespace bitpit;

namespace mesh_adaptation {

void markCellsForRefinement(VolOctree &mesh)
{
    mesh.markCellForRefinement(0);
}

void meshAdaptation(VolOctree &mesh, ScalarStorage<std::size_t> &previousIDs,
                    ScalarStorage<std::size_t> &currentIDs)
{
    std::vector<adaption::Info> adaptionData;
    markCellsForRefinement(mesh);
    bool trackAdaptation = true;
    adaptionData = mesh.adaptionPrepare(trackAdaptation);

    bool squeeshPatchStorage = false;
    adaptionData = mesh.adaptionAlter(trackAdaptation, squeeshPatchStorage);

    for (const adaption::Info &adaptionInfo : adaptionData) {
        // Consider only cell refinements
        if (adaptionInfo.entity != adaption::Entity::ENTITY_CELL) {
            continue;
        } else if (adaptionInfo.type != adaption::TYPE_REFINEMENT) {
            continue;
        }
        // Assign data to children
        long parentId = adaptionInfo.previous.front();
        for (long currentId : adaptionInfo.current) {
            previousIDs.push_back(parentId);
            currentIDs.push_back(currentId);
        }
    }

    mesh.adaptionCleanup();
    mesh.write();
}

}
