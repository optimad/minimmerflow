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
#include "test.hpp"


using namespace bitpit;

namespace adaptation {
/*
 * Marks in VolOCtree object the cells to be refined
 * \param reference to the mesh object
 * \param time step
 * \param initial refinement level of cells
 * \param maximum allowed refinement level of cells
 * \param[out] reference to the number of cells to be refined
 * \param[out] reference to the mass-center of selection-cube
*/
void markCellsForRefinement(VolOctree &mesh, const double time, int &nCellsToBeRefined, const std::array<double,2> initialOrigin,
                            int &initialRefinementLevel, int &maxRefinementLevel)
{
    // GLOBAL
 // int nCellsToBeRefined = 0;
 // for (int iter = 0; iter < mesh.getCellCount(); iter++) {
 //     mesh.markCellForRefinement(iter);
 //     nCellsToBeRefined++;
 // }
    // Create refinement cube which tracks vortex_xy
//  const double cubeLength = 6;

    std::array<double,2> newOrigin;
    std::array<double,2> cubeVelocity({1, 1});
    newOrigin[0] = initialOrigin[0] + time * cubeVelocity[0];
    newOrigin[1] = initialOrigin[1] + time * cubeVelocity[1];

//  std::vector<double> lowerLimits(2,0);
//  std::vector<double> upperLimits(2,0);
//  lowerLimits[0] = newOrigin[0] - 0.5 * cubeLength;
//  lowerLimits[1] = newOrigin[1] - 0.5 * cubeLength;
//  upperLimits[0] = newOrigin[0] + 0.5 * cubeLength;
//  upperLimits[1] = newOrigin[1] + 0.5 * cubeLength;

    nCellsToBeRefined = 0;
    if (initialRefinementLevel == -1) {
        initialRefinementLevel = mesh.getCellLevel(0);
        maxRefinementLevel += initialRefinementLevel - 1;
    }
    double waveEdge = 1.54 * time;
    std::cout << "waveEdge  " << waveEdge  << std::endl;

    for (auto cell : mesh.getCells()) {
        long cellId = cell.getId();
        if (mesh.getCellLevel(cellId) <= maxRefinementLevel) {
            // Create inner cycle
            double x = mesh.evalCellCentroid(cellId)[0];
            double y = mesh.evalCellCentroid(cellId)[1];
            double sqrtDistance = (x-newOrigin[0]) * (x-newOrigin[0]) + (y-newOrigin[1]) * (y-newOrigin[1]);
//          if (lowerLimits[0] <= x && x <= upperLimits[0] && lowerLimits[1] <=y && y <= upperLimits[1] ) {
            if (sqrtDistance <= 9) { // sqrt(9) is the radius... sorry, for the time being it's hard-coded to death
                mesh.markCellForRefinement(cellId);
                nCellsToBeRefined++;
//          } else if ( (sqrtDistance > 2.25 && sqrtDistance <= 9) && (mesh.getCellLevel(cellId) <= maxRefinementLevel - 1)) {
//              mesh.markCellForRefinement(cellId);
//              nCellsToBeRefined++;
            } else if ( (sqrtDistance > 9 && sqrtDistance <= 16) && (mesh.getCellLevel(cellId) <= maxRefinementLevel - 2)) {
                mesh.markCellForRefinement(cellId);
                nCellsToBeRefined++;
            } else if
            (
//              (sqrtDistance >= 48 && sqrtDistance <= 50)
//           && (mesh.getCellLevel(cellId) <= maxRefinementLevel - 2)
//              (sqrtDistance >= 47)
//           && (sqrtDistance <= 51)
//           && (time >= 2.325)
//           && (mesh.getCellLevel(cellId) >= maxRefinementLevel - 3))
//              (waveEdge > 4 && sqrtDistance <= 51)
                (waveEdge > 4)
             && (sqrtDistance <= waveEdge * waveEdge)
             && (mesh.getCellLevel(cellId) <= maxRefinementLevel - 2))
            {
                mesh.markCellForRefinement(cellId);
                nCellsToBeRefined++;
            }
        }
    }
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
 * \param time step
 * \param initial refinement level of cells
 * \param maximum allowed refinement level of cells
 * \param[out] field of RHS-of-equations at cells
 * \param[out] field of conservatives at cells
 * \param[out] field of primitives at cells
 * \param[out] field of conservativesWork at cells
 */
void meshAdaptation(VolOctree &mesh, std::vector<std::vector<double>> &parentCellRHS, std::vector<std::vector<double>> &parentCellConservatives,
                    ScalarPiercedStorageCollection<double> &cellRHS, ScalarPiercedStorageCollection<double> &cellConservatives, const double time,
                    const std::array<double,2> initialOrigin, int &initialRefinementLevel, int &maxRefinementLevel)
{
    std::vector<adaption::Info> adaptionData;
    int nCellsToBeRefined = 0;
    markCellsForRefinement(mesh, time, nCellsToBeRefined, initialOrigin, initialRefinementLevel, maxRefinementLevel);
    std::vector<long> oldLocalIDs;

//  int localID(0);
//  for (int iter = 0; iter < nCellsToBeRefined; iter++) {
//      for (int iter = 0; iter < 4; iter++) {
//          oldLocalIDs[4 * localID + iter] = localID;
//          std::cout << "oldLocalIDs[" << 4 * localID + iter << "] = " << localID << std::endl;
//      }
//      localID++;
//  }
    bool trackAdaptation = true;
    adaptionData = mesh.adaptionPrepare(trackAdaptation);

    const std::size_t cellSize = mesh.getCellCount();

    long localID(0);

    for (const adaption::Info &adaptionInfo : adaptionData) {
        // Consider only cell refinements
        if (adaptionInfo.entity != adaption::Entity::ENTITY_CELL) {
            continue;
        } else if (adaptionInfo.type != adaption::TYPE_REFINEMENT) {
            continue;
        }

        // Save parent data
        for (long parentId : adaptionInfo.previous) {
            oldLocalIDs.push_back(localID);
         // for (int iter = 0; iter < 4; iter++) {
         //     oldLocalIDs.push_back(localID);
         // }
            for (int iField = 0; iField < N_FIELDS; iField++) {
                parentCellRHS[iField].push_back(cellRHS[iField].at(parentId));
                parentCellConservatives[iField].push_back(cellConservatives[iField].at(parentId));
            }
            localID++;
        }
    }

    bool squeeshPatchStorage = false;
    adaptionData = mesh.adaptionAlter(trackAdaptation, squeeshPatchStorage);

    int count(0);
    for (const adaption::Info &adaptionInfo : adaptionData) {
        // Consider only cell refinements
        if (adaptionInfo.entity != adaption::Entity::ENTITY_CELL) {
            continue;
        } else if (adaptionInfo.type != adaption::TYPE_REFINEMENT) {
            continue;
        }

        // Assign data to children
        long parentId = adaptionInfo.previous.front();
        int localID = oldLocalIDs[count];
        for (long currentId : adaptionInfo.current) {
            for (int iField = 0; iField < N_FIELDS; iField++) {
                cellRHS[iField].set(currentId, parentCellRHS[iField][localID]);
                cellConservatives[iField].set(currentId, parentCellConservatives[iField][localID]);
            }
        }
        count++;
    }

    mesh.adaptionCleanup();
//  mesh.write();

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
