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

#include "adaptation.hpp"

using namespace bitpit;

namespace adaptation {

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

/*
 * Map solution from previous mesh to new one
 *
 * \param ids of pre-adaptation mesh
 * \param ids of post-adaptation mesh
 * \param[out] mapped field (on post-adaptation mesh)
 */
void mapField(ScalarStorage<std::size_t> &previousIDs,
              ScalarStorage<std::size_t> &currentIDs,
              ScalarPiercedStorage<double> &field)
{
#if ENABLE_CUDA
    cuda_mapField(previousIDs.data(), currentIDs.data(), field.data());
#else
    cpu_mapField(previousIDs, currentIDs, field);
#endif
}

/*
 * Map solution from previous mesh to new one on CPU
 *
 * \param ids of pre-adaptation mesh
 * \param ids of post-adaptation mesh
 * \param[out] mapped field (on post-adaptation mesh)
 */
void cpu_mapField(ScalarStorage<std::size_t> &previousIDs,
                  ScalarStorage<std::size_t> &currentIDs,
                  ScalarPiercedStorage<double> &field)
{
    for (int i = 0; i < previousIDs.size(); i++) {
        const long previousID = previousIDs[i];
//      double parentValue = field.at(previousId);
        const long currentID = currentIDs[i];
        //  TODO
//      field[currentID] = parentValue;
    }
}

/*
 * Map solution from previous mesh to new one for flow variables
 *
 * \param ids of pre-adaptation mesh
 * \param ids of post-adaptation mesh
 * \param computationInfo object
 * \param[out] field of RHS-of-equations at cells
 * \param[out] field of conservatives at cells
 * \param[out] field of primitives at cells
 * \param[out] field of conservativesWork at cells
 */
void mapFields(ScalarStorage<std::size_t> &previousIDs,
               ScalarStorage<std::size_t> &currentIDs,
               ScalarPiercedStorage<double> &cellRHS,
               ScalarPiercedStorage<double> &cellConservatives,
               ScalarPiercedStorage<double> &cellPrimitives,
               ScalarPiercedStorage<double> &cellConservativesWork,
               ComputationInfo &computationInfo)
{
    computationInfo.MeshGeometricalInfo::cuda_finalize();
    mapField(previousIDs, currentIDs, cellRHS);
    mapField(previousIDs, currentIDs, cellConservatives);
    mapField(previousIDs, currentIDs, cellPrimitives);
    mapField(previousIDs, currentIDs, cellConservativesWork);
}


}