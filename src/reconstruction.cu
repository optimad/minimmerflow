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

#include "reconstruction.hcu"

/*!
 * Gets a pointer to the cell polynomial CUDA data storage.
 *
 * \result A pointer to the cell polynomial CUDA data storage.
 */
double ** ReconstructionCalculator::cuda_getCellPolynomialDevData()
{
    return m_cellPolynomials.cuda_deviceCollectionData();
}

/*!
 * Gets a constant pointer to the cell polynomial CUDA data storage.
 *
 * \result A constant pointer to the cell polynomial CUDA data storage.
 */
const double * const * ReconstructionCalculator::cuda_getCellPolynomialDevData() const
{
    return m_cellPolynomials.cuda_deviceCollectionData();
}

/*!
 * Initialize CUDA operations.
 */
void ReconstructionCalculator::cuda_initialize()
{
    // Allocate device memory
    m_cellSupportSizes.cuda_allocateDevice();
    m_cellSupportOffsets.cuda_allocateDevice();
    m_cellSupportRawIds.cuda_allocateDevice();
    m_cellKernelWeights.cuda_allocateDevice();
    m_cellPolynomials.cuda_allocateDevice();

    // Copy data to the device
    m_cellSupportSizes.cuda_updateDevice();
    m_cellSupportOffsets.cuda_updateDevice();
    m_cellSupportRawIds.cuda_updateDevice();
    m_cellKernelWeights.cuda_updateDevice();
}

/*!
 * Finalize CUDA operations.
 */
void ReconstructionCalculator::cuda_finalize()
{
    // Deallocate device memory
    m_cellSupportSizes.cuda_freeDevice();
    m_cellSupportOffsets.cuda_freeDevice();
    m_cellSupportRawIds.cuda_freeDevice();
    m_cellKernelWeights.cuda_freeDevice();
    m_cellPolynomials.cuda_freeDevice();
}

/*!
 * Updates cell reconstruction polynomials.
 *
 * \param cellConservatives are the cell conservative values
 */
void ReconstructionCalculator::cuda_updateCellPolynomials(const ScalarPiercedStorageCollection<double> &cellConservatives)
{
    //
    // Initialization
    //
    int devOrder     = getOrder();
    int devDimension = getDimension();

    const ScalarStorage<std::size_t> &reconstructedCellRawIds = m_computationInfo.getReconstructedCellRawIds();

    const std::size_t nReconstructedCells    = reconstructedCellRawIds.size();
    const std::size_t *devCellRawIds         = reconstructedCellRawIds.cuda_deviceData();
    const std::size_t *devCellSupportSizes   = m_cellSupportSizes.cuda_deviceData();
    const std::size_t *devCellSupportOffsets = m_cellSupportOffsets.cuda_deviceData();
    const long *devCellSupportRawIds         = m_cellSupportRawIds.cuda_deviceData();
    const double *devCellKernelWeigths       = m_cellKernelWeights.cuda_deviceData();

    const double * const *devCellConservatives = cellConservatives.cuda_deviceCollectionData();

    double **devCellPolynomials = cuda_getCellPolynomialDevData();
    int devCellPolynomialsBlockSize = getCellPolynomials().getFieldCount();

    //
    // Device properties
    //
    int device;
    cudaGetDevice(&device);

    int nMultiprocessors;
    cudaDeviceGetAttribute(&nMultiprocessors, cudaDevAttrMultiProcessorCount, device);

    //
    // Process cells
    //

    // Get block information
    const int UNIFORM_BLOCK_SIZE  = 256;
    const int nCellBlocks = 32 * nMultiprocessors;

    // Evaluate fluxes
    reconstruction::dev_updateCellPolynomials<UNIFORM_BLOCK_SIZE><<<nCellBlocks, UNIFORM_BLOCK_SIZE>>>
    (
        devDimension, devOrder, nReconstructedCells,
        devCellRawIds, devCellSupportSizes, devCellSupportOffsets, devCellSupportRawIds,
        devCellConservatives, devCellKernelWeigths, devCellPolynomials, devCellPolynomialsBlockSize
    );
}

