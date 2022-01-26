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
#include "constants.hpp"

using namespace bitpit;

namespace adaptation {
/*
 * Hardcoded cell-selection for mesh refinement.
 * TODO: See if this can be done in a better way.
*/
void markCellsForRefinement(VolOctree &mesh)
{
    mesh.markCellForRefinement(0);
}

/*
 * Adapt mesh. In case of running on CPUs, map solution from previous
 * mesh to new one. In case of GPUs, mapping is performing on them, so the
 * correspondance between the parent and the new IDs is stored now on CPUs,
 * so that it is copied to GPUs later.
 *
 * \param ids of cells to be refined, at the stabe befor the mesh adaptation
 * \param ids of post-adaptation mesh
 * \param parent field of RHS-of-equations at cells
 * \param parent field of conservatives at cells
 * \param parent field of primitives at cells
 * \param parent field of conservativesWork at cells
 * \param[out] field of RHS-of-equations at cells
 * \param[out] field of conservatives at cells
 * \param[out] field of primitives at cells
 * \param[out] field of conservativesWork at cells
 */
void meshAdaptation(VolOctree &mesh, ScalarStorage<std::size_t> &parentIDs,
                    ScalarStorage<std::size_t> &currentIDs,
                    ScalarStorageCollection<double> &parentCellRHS,
                    ScalarStorageCollection<double> &parentCellConservatives,
                    ScalarPiercedStorageCollection<double> &cellRHS,
                    ScalarPiercedStorageCollection<double> &cellConservatives)
{
    std::vector<adaption::Info> adaptionData;
    markCellsForRefinement(mesh);
    bool trackAdaptation = true;
    adaptionData = mesh.adaptionPrepare(trackAdaptation);

    const std::size_t cellSize = mesh.getCellCount();

#if ENABLE_CUDA
    ScalarStorage<std::size_t> tempParentIDs;
#endif
    for (const adaption::Info &adaptionInfo : adaptionData) {
        // Consider only cell refinements
        if (adaptionInfo.entity != adaption::Entity::ENTITY_CELL) {
            continue;
        } else if (adaptionInfo.type != adaption::TYPE_REFINEMENT) {
            continue;
        }

        // Save parent data
        for (long parentId : adaptionInfo.previous) {
            long parentRawId = mesh.getVertex(parentId).getId();
#if ENABLE_CUDA
            tempParentIDs.push_back(parentRawId);
#else
            for (int iField = 0; iField < N_FIELDS; iField++) {
                parentCellRHS[parentRawId * N_FIELDS + iField] = cellRHS.at(parentId, iField);
                parentCellConservatives[parentRawId * N_FIELDS + iField] = cellConservatives.at(parentId, iField);
            }
#endif
        }
    }
#if ENABLE_CUDA
    tempParentIDs.cuda_allocateDevice();
    tempParentIDs.cuda_updateDevice();
    for (int k = 0; k < N_FIELDS; k++) {
        parentCellRHS[k].resize(tempParentIDs.size());
        parentCellConservatives[k].resize(tempParentIDs.size());
    }
    parentCellRHS.cuda_allocateDevice();
    parentCellConservatives.cuda_allocateDevice();
    parentCellRHS.cuda_updateDevice();
    parentCellConservatives.cuda_updateDevice();
#endif

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
            long parentRawId = mesh.getVertex(parentId).getId();
            long currentRawId = mesh.getVertex(currentId).getId();
#if ENABLE_CUDA
            currentIDs.push_back(currentRawId);
            parentIDs.push_back(parentRawId);
#else
            for (int iField = 0; iField < N_FIELDS; iField++) {
                cellRHS.set(currentId, iField, parentCellRHS[N_FIELDS * parentRawId + iField];
                cellConservatives.set(currentId, iField, parentCellConservatives[N_FIELDS * parentRawId + iField];
            }
#endif
        }
    }

#if ENABLE_CUDA
    cuda_storeParentField(tempParentIDs, parentCellRHS, cellRHS);
    cuda_storeParentField(tempParentIDs, parentCellConservatives, cellConservatives);

    tempParentIDs.cuda_freeDevice();
#endif
    mesh.adaptionCleanup();
    mesh.write();
}

/*
 * Map solution from previous mesh to new one, on GPUs.
 * TODO: Implement with streams, since we do this independently for each of the
 e four containers and we can hide data-transfer
 *
 * \param ids of pre-adaptation mesh
 * \param ids of post-adaptation mesh
 * \param[out] mapped field (on post-adaptation mesh)
 */
void mapField(ScalarStorage<std::size_t> &parentIDs,
              ScalarStorage<std::size_t> &currentIDs,
              ScalarStorageCollection<double> &parentField,
              ScalarPiercedStorageCollection<double> &field)
{
    // Perform mapping
    cuda_mapField(parentIDs, currentIDs, parentField, field);
}


/*
 * Map solution from previous mesh to new one for flow variables, to be used
 * when running with CUDA on GPUs
 *
 * \param ids of pre-adaptation mesh
 * \param ids of post-adaptation mesh
 * \param computationInfo object
 * \param[out] field of RHS-of-equations at cells
 * \param[out] field of conservatives at cells
 * \param[out] field of primitives at cells
 * \param[out] field of conservativesWork at cells
 */
void mapFields(ScalarStorage<std::size_t> &parentIDs,
               ScalarStorage<std::size_t> &currentIDs,
               ScalarStorageCollection<double> &parentCellRHS,
               ScalarStorageCollection<double> &parentCellConservatives,
               ScalarPiercedStorageCollection<double> &cellRHS,
               ScalarPiercedStorageCollection<double> &cellConservatives)
{
#if ENABLE_CUDA
    mapField(parentIDs, currentIDs, parentCellRHS, cellRHS);
    mapField(parentIDs, currentIDs, parentCellConservatives, cellConservatives);
#else
    BITPIT_UNUSED(parentIDs);
    BITPIT_UNUSED(currentIDs);

    BITPIT_UNUSED(parentCellRHS);
    BITPIT_UNUSED(parentCellConservatives);

    BITPIT_UNUSED(cellRHS);
    BITPIT_UNUSED(cellConservatives);
#endif
}

}
