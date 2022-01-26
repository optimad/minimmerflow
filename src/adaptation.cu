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

#include "adaptation.hcu"

namespace adaptation {


void cuda_storeParentField(ScalarStorage<std::size_t> &parentIDs,
                           ScalarStorageCollection<double> &parentField,
                           ScalarPiercedStorageCollection<double> &field)
{
    const int BLOCK_SIZE = 256;
    int nBlocks = (parentIDs.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dev_storeParentField<<<nBlocks, BLOCK_SIZE>>>(parentIDs.cuda_deviceData(),
                                                  parentField.cuda_deviceCollectionData(),
                                                  field.cuda_deviceCollectionData(),
                                                  parentIDs.cuda_deviceDataSize());
}


/*
 * Map solution from previous mesh to new one on GPU
 *
 * \param ids of pre-adaptation mesh
 * \param ids of post-adaptation mesh
 * \param[out] mapped field (on post-adaptation mesh)
 */
void cuda_mapField(ScalarStorage<std::size_t> &parentIDs,
                   ScalarStorage<std::size_t> &currentIDs,
                   ScalarStorageCollection<double> &parentField,
                   ScalarPiercedStorageCollection<double> &field)
{

    const std::size_t currentIDsSize = currentIDs.cuda_deviceDataSize();

    const int BLOCK_SIZE = 256;
    int nBlocks = (currentIDsSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dev_mapField<<<nBlocks, BLOCK_SIZE>>>(parentIDs.cuda_deviceData(),
                                          currentIDs.cuda_deviceData(),
                                          parentField.cuda_deviceCollectionData(),
                                          field.cuda_deviceCollectionData(),
                                          currentIDsSize);
}

/*
 * Map solution from previous mesh to new one on GPU
 *
 * \param ids of pre-adaptation mesh
 * \param ids of post-adaptation mesh
 * \param[out] mapped field (on post-adaptation mesh)
 */
__global__ void dev_mapField(std::size_t *parentIDs, std::size_t *currentIDs,
                             double **parentField, double **field,
                             const std::size_t idsSize)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= idsSize) {
        return;
    };

    const std::size_t parentId = parentIDs[i];
    const std::size_t currentId = currentIDs[i];

    for (int k = 0; k < N_FIELDS; k++) {
        field[k][currentId] = parentField[k][parentId];
    }
}


__global__ void dev_storeParentField(const size_t *parentIDs, double **parentField,
                                     double **field, const size_t parentIDsSize)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= parentIDsSize) {
        return;
    }

    std::size_t position = parentIDs[i];
    for (int k = 0; k < N_FIELDS; k++) {
        parentField[k][position] = field[k][position];
    }
}


}
