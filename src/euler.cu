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

#include "euler.hcu"
#include "utils_cuda.hpp"

#include <float.h>

#define uint64  unsigned long long

namespace euler {

double *devMaxEig;

/**
 * Compute the maximum of two double-precision floating point values using an
 * atomic operation.
 *
 * \param value is the value that is compared to the reference in order to
 * determine the maximum
 * \param[in,out] maxValue is the address of the reference value which might
 * get updated with the maximum
 */
__device__ void dev_atomicMax(const double value, double * const maxValue)
{
    if (*maxValue >= value) {
        return;
    }

    uint64 oldMaxValue = *((uint64 *) maxValue);
    uint64 assumedMaxValue;
    do {
        assumedMaxValue = oldMaxValue;
        if (__longlong_as_double(assumedMaxValue) >= value) {
            break;
        }

        oldMaxValue = atomicCAS((uint64 *) maxValue, assumedMaxValue, __double_as_longlong(value));
    } while (assumedMaxValue != oldMaxValue);
}

/**
 * Compute the maximum of double-precision floating point values.
 *
 * \param value is the value that is compared in order to determine the maximum
 * \param nElements is the number of the elements that will be compared
 * \param[in,out] workspace is a working array whose size should be at
 * least equal to the number of blocks
 * \param[in,out] maxValue is the address of the reference value which might
 * get updated with the maximum
 */
__device__ void dev_reduceMax(const double value, const size_t nElements,
                              double *workspace, double *maxValue)
{
    // Get thread and global ids
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;

    // Put thread value in the array that stores block values
    if (gid < nElements) {
        workspace[tid] = value;
    } else {
        workspace[tid] = - DBL_MAX;
    }
    __syncthreads();

    // Evaluate the maximum of each block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < nElements) {
            workspace[tid] = max(workspace[tid], workspace[tid + s]);
        }
        __syncthreads();
    }

    // Evaluate the maximum among different blocks
    if (tid == 0) {
        dev_atomicMax(workspace[0], maxValue);
    }
}

/*!
 * Initialize CUDA computation
 */
void cuda_initialize()
{
    CUDA_ERROR_CHECK(cudaMalloc((void **) &devMaxEig, 1 * sizeof(double)));
}

/*!
 * Finalize CUDA computation
 */
void cuda_finalize()
{
    CUDA_ERROR_CHECK(cudaFree(devMaxEig));
}

/*!
 * Reset the RHS.
 *
 * \param[in,out] rhs is the RHS that will be reset
 */
void cuda_resetRHS(ScalarPiercedStorageCollection<double> *cellsRHS)
{
    cellsRHS->cuda_fillDevice(0.);
}

/*!
 * Update cell RHS.
 *
 * \param problemType is the problem type
 * \param computationInfo are the computation information
 * \param order is the order
 * \param interfaceBCs is the boundary conditions storage
 * \param cellConservatives are the cell conservative values
 * \param[out] cellsRHS on output will containt the RHS
 * \param[out] maxEig on putput will containt the maximum eigenvalue
 */
void cuda_updateRHS(problem::ProblemType problemType, ComputationInfo &computationInfo,
                    const int order, const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
                    const ScalarPiercedStorageCollection<double> &cellConservatives, ScalarPiercedStorageCollection<double> *cellsRHS, double *maxEig)
{
    //
    // Initialization
    //
    const double * devInterfaceAreas            = computationInfo.cuda_getInterfaceAreaDevData();
    const double * const *devInterfaceCentroids = computationInfo.cuda_getInterfaceCentroidDevData();
    const double * const *devInterfaceNormals   = computationInfo.cuda_getInterfaceNormalDevData();

    CUDA_ERROR_CHECK(cudaMemset(devMaxEig, 0., 1 * sizeof(double)));

    double **devCellsRHS = cellsRHS->cuda_deviceCollectionData();

    const double * const *devCellConservatives = cellConservatives.cuda_deviceCollectionData();

    int devProblemType = static_cast<int>(problemType);

    //
    // Device properties
    //
    int device;
    cudaGetDevice(&device);

    int nMultiprocessors;
    cudaDeviceGetAttribute(&nMultiprocessors, cudaDevAttrMultiProcessorCount, device);

    //
    // Process uniform interfaces
    //
    const ScalarStorage<std::size_t> &uniformInterfaceRawIds      = computationInfo.getSolvedUniformInterfaceRawIds();
    const ScalarStorage<std::size_t> &uniformInterfaceOwnerRawIds = computationInfo.getSolvedUniformInterfaceOwnerRawIds();
    const ScalarStorage<std::size_t> &uniformInterfaceNeighRawIds = computationInfo.getSolvedUniformInterfaceNeighRawIds();

    const std::size_t nSolvedUniformInterfaces   = uniformInterfaceRawIds.size();
    const std::size_t *devUniformInterfaceRawIds = uniformInterfaceRawIds.cuda_deviceData();
    const std::size_t *devUniformOwnerRawIds     = uniformInterfaceOwnerRawIds.cuda_deviceData();
    const std::size_t *devUniformNeighRawIds     = uniformInterfaceNeighRawIds.cuda_deviceData();

    // Get block information
    const int UNIFORM_BLOCK_SIZE  = 256;
    const int UNIFORM_SHARED_SIZE = 3 * N_FIELDS * UNIFORM_BLOCK_SIZE * sizeof(double);
    int nUniformBlocks = 32 * nMultiprocessors;

    // Evaluate fluxes
    dev_uniformUpdateRHS<UNIFORM_BLOCK_SIZE><<<nUniformBlocks, UNIFORM_BLOCK_SIZE, UNIFORM_SHARED_SIZE>>>
    (
        nSolvedUniformInterfaces, order,
        devUniformInterfaceRawIds, devInterfaceAreas,devInterfaceNormals, devInterfaceCentroids,
        devUniformOwnerRawIds, devUniformNeighRawIds, devCellConservatives,
        devCellsRHS, devMaxEig
    );

    //
    // Process boundary interfaces
    //
    const ScalarStorage<std::size_t> &boundaryInterfaceRawIds      = computationInfo.getSolvedBoundaryInterfaceRawIds();
    const ScalarStorage<std::size_t> &boundaryInterfaceFluidRawIds = computationInfo.getSolvedBoundaryInterfaceFluidRawIds();
    const ScalarStorage<int> &boundaryInterfaceSigns               = computationInfo.getSolvedBoundaryInterfaceSigns();

    const std::size_t nBoundaryInterfaces         = boundaryInterfaceRawIds.size();
    const std::size_t *devBoundaryInterfaceRawIds = boundaryInterfaceRawIds.cuda_deviceData();
    const std::size_t *devBoundaryFluidRawIds     = boundaryInterfaceFluidRawIds.cuda_deviceData();
    const int *devBoundaryInterfaceSigns          = boundaryInterfaceSigns.cuda_deviceData();
    const int *devBoundaryInterfaceBCs            = solvedBoundaryInterfaceBCs.cuda_deviceData();

    // Get block information
    const int BOUNDARY_BLOCK_SIZE  = 256;
    const int BOUNDARY_SHARED_SIZE = 3 * N_FIELDS * BOUNDARY_BLOCK_SIZE * sizeof(double);;
    int nBoundaryBlocks = 32 * nMultiprocessors;

    // Evaluate fluxes
    dev_boundaryUpdateRHS<BOUNDARY_BLOCK_SIZE><<<nBoundaryBlocks, BOUNDARY_BLOCK_SIZE, BOUNDARY_SHARED_SIZE>>>
    (
        nBoundaryInterfaces, devProblemType, order,
        devBoundaryInterfaceRawIds, devInterfaceAreas, devInterfaceNormals, devInterfaceCentroids,
        devBoundaryFluidRawIds, devCellConservatives,
        devBoundaryInterfaceSigns, devBoundaryInterfaceBCs,
        devCellsRHS, devMaxEig
    );

    //
    // Update host memory
    //
    cellsRHS->cuda_updateHost();
    CUDA_ERROR_CHECK(cudaMemcpy(maxEig, devMaxEig, sizeof(double), cudaMemcpyDeviceToHost));
}

}
