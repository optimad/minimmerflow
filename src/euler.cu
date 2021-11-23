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

#include "containers.cu"
#include "euler.hcu"
#include "problem.hcu"
#include "reconstruction.hcu"
#include "utils_cuda.hpp"

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
 * \param[in,out] maxValue is the address of the reference value which might
 * get updated with the maximum
 */
__device__ void dev_reduceMax(const double value, const size_t nElements, double *maxValue)
{
    extern __shared__ double blockValues[];

    // Get thread and global ids
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;

    // Put thread value in the array that stores block values
    blockValues[tid] = value;
    __syncthreads();

    // Evaluate the maximum of each block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < nElements) {
            blockValues[tid] = max(blockValues[tid], blockValues[tid + s]);
        }
        __syncthreads();
    }

    // Evaluate the maximum among different blocks
    if (tid == 0) {
        dev_atomicMax(blockValues[0], maxValue);
    }
}

/*!
 * Calculates the conservative fluxes for a perfect gas.
 *
 * \param conservative is the conservative state
 * \param n is the normal direction
 * \param[out] fluxes on output will contain the conservative fluxes
 * \param[out] lambda on output will contain the maximum eigenvalue
 */
__device__ void dev_evalFluxes(const DeviceCollectionDataConstCursor<double> &conservative,
                               const double *normal,
                               double *fluxes, double *lambda)
{
    // Compute variables
    double primitive[N_FIELDS];
    ::utils::dev_conservative2primitive(conservative, primitive);

    double rho = conservative[DEV_FID_RHO];
    double p   = primitive[DEV_FID_P];

    double nx = normal[0];
    double ny = normal[1];
    double nz = normal[2];

    double u = primitive[DEV_FID_U];
    double v = primitive[DEV_FID_V];
    double w = primitive[DEV_FID_W];

    double un = ::utils::dev_normalVelocity(u, v, w, nx, ny, nz);

    double T = primitive[DEV_FID_T];
    double a = std::sqrt(DEV_GAMMA * T);

    // Evaluate maximum eigenvalue
    *lambda = std::abs(un) + a;

    // Maxx flux
    double massFlux = rho * un;

    // Energy flux
    double eto = p / (DEV_GAMMA - 1.) + 0.5 * rho * (u * u + v * v + w * w);

    fluxes[DEV_FID_EQ_E] = un * (eto + p);

    // Continuity flux
    fluxes[DEV_FID_EQ_C] = massFlux;

    // Momentum flux
    fluxes[DEV_FID_EQ_M_X] = massFlux * u + p * nx;
    fluxes[DEV_FID_EQ_M_Y] = massFlux * v + p * ny;
    fluxes[DEV_FID_EQ_M_Z] = massFlux * w + p * nz;

    // Check consistency of fluid dynamic state
    if (p < 0.) {
        printf("***** Negative pressure (%f) in flux computation!\n", p);
    }

    if (rho < 0.) {
       printf("***** Negative density (%f) in flux computation!\n", rho);
    }
}

/*!
 * Computes approximate Riemann solver (Local Lax Friedrichs) for compressible perfect gas.
 *
 * \param conservativeL is the left conservative state
 * \param conservativeR is the right conservative state
 * \param n is the normal
 * \param[out] fluxes on output will contain the conservative fluxes
 * \param[out] lambda on output will contain the maximum eigenvalue
 */
__device__ void dev_evalSplitting(const DeviceCollectionDataConstCursor<double> &conservativeL, const DeviceCollectionDataConstCursor<double> &conservativeR,
                                  const double *normal, double *fluxes, double *lambda)
{
    // Fluxes
    double fL[N_FIELDS];
    double lambdaL;
    dev_evalFluxes(conservativeL, normal, fL, &lambdaL);

    double fR[N_FIELDS];
    double lambdaR;
    dev_evalFluxes(conservativeR, normal, fR, &lambdaR);

    // Splitting
    *lambda = max(lambdaR, lambdaL);

    for (int k = 0; k < N_FIELDS; ++k) {
        fluxes[k] = 0.5 * ((fR[k] + fL[k]) - (*lambda) * (conservativeR[k] - conservativeL[k]));
    }
}

/*!
 * Computes the boundary values for the free flow BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner innerValues values
 * \param[out] boundaryValues are the boundary values
 */
__device__ void dev_evalFreeFlowBCValues(const double *point, const double *normal,
                                         const double *info, const DeviceCollectionDataConstCursor<double> &innerValues,
                                         DeviceCollectionDataCursor<double> *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);
    BITPIT_UNUSED(info);

    for (int i = 0; i < N_FIELDS; ++i) {
        (*boundaryValues)[i] = innerValues[i];
    }
}

/*!
 * Computes the boundary values for the reflecting BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner innerValues values
 * \param[out] boundaryValues are the boundary values
 */
__device__ void dev_evalReflectingBCValues(const double *point, const double *normal,
                                           const double *info, const DeviceCollectionDataConstCursor<double> &innerValues,
                                           DeviceCollectionDataCursor<double> *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(info);

    double primitive[N_FIELDS];
    ::utils::dev_conservative2primitive(innerValues, primitive);

    double nx = normal[0];
    double ny = normal[1];
    double nz = normal[2];

    double u = primitive[DEV_FID_U];
    double v = primitive[DEV_FID_V];
    double w = primitive[DEV_FID_W];

    double un = ::utils::dev_normalVelocity(u, v, w, nx, ny, nz);

    primitive[FID_U] -= 2 * un * nx;
    primitive[FID_V] -= 2 * un * ny;
    primitive[FID_W] -= 2 * un * nz;

    ::utils::dev_primitive2conservative(primitive, boundaryValues);
}

/*!
 * Computes the boundary values for the wall BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner innerValues values
 * \param[out] boundaryValues are the boundary values
 */
__device__ void dev_evalWallBCValues(const double *point, const double *normal,
                                     const double *info, const DeviceCollectionDataConstCursor<double> &innerValues,
                                     DeviceCollectionDataCursor<double> *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(info);

    dev_evalReflectingBCValues(point, normal, info, innerValues, boundaryValues);
}

/*!
 * Computes the boundary values for the dirichlet BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner innerValues values
 * \param[out] boundaryValues are the boundary values
 */
__device__ void dev_evalDirichletBCValues(const double *point, const double *normal,
                                          const double *info, const DeviceCollectionDataConstCursor<double> &innerValues,
                                          DeviceCollectionDataCursor<double> *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);
    BITPIT_UNUSED(innerValues);

    ::utils::dev_primitive2conservative(info, boundaryValues);
}

/*!
 * Computes the boundary values for the specified interface.
 *
 * \param problemType is the type of problem being solved
 * \param BCType is the type of boundary condition to apply
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param innerValues are the inner innerValues values
 * \param[out] boundaryValues are the boundary values
 */
__device__ void dev_evalInterfaceBCValues(int problemType, int BCType,
                                          const double *point, const double *normal,
                                          const DeviceCollectionDataConstCursor<double> &innerValues,
                                          DeviceCollectionDataCursor<double> *boundaryValues)
{
    double info[BC_INFO_SIZE];
    problem::dev_getBorderBCInfo(problemType, BCType, point, normal, info);

    switch (BCType)
    {
        case BC_FREE_FLOW:
            dev_evalFreeFlowBCValues(point, normal, info, innerValues, boundaryValues);
            break;

        case BC_REFLECTING:
            dev_evalReflectingBCValues(point, normal, info, innerValues, boundaryValues);
            break;

        case BC_WALL:
            dev_evalWallBCValues(point, normal, info, innerValues, boundaryValues);
            break;

        case BC_DIRICHLET:
            dev_evalDirichletBCValues(point, normal, info, innerValues, boundaryValues);
            break;

    }
}

/*!
 * Evaluate cell values on interface centroids.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceCentroids are the centroid of the interfaces
 * \param cellRawIds are the raw ids of the cells
 * \param cellValues are the cell values
 * \param order is the reconstruction order
 * \param[out] interfaceValues are the interface values
 */
__global__ void dev_evalInterfaceValues(std::size_t nInterfaces, const std::size_t *interfaceRawIds, const double * const *interfaceCentroids,
                                        const std::size_t *cellRawIds, const double * const *cellValues,
                                        int order, double **interfaceValues)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];
    double interfaceCentroid[3] = {interfaceCentroids[0][interfaceRawId], interfaceCentroids[1][interfaceRawId], interfaceCentroids[2][interfaceRawId]};

    // Cell information
    const std::size_t cellRawId = cellRawIds[i];
    DeviceCollectionDataConstCursor<double> meanValues(cellValues, cellRawId);

    // Reconstruct interface values
    DeviceCollectionDataCursor<double> reconstructedValues(interfaceValues, i);
    reconstruction::dev_eval(order, interfaceCentroid, meanValues, &reconstructedValues);
}

/*!
 * Evaluate boundary conditions on interface centroids.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceBCs are the boundary conditions associated with the
 * interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param interfaceCentroids are the centroid of the interfaces
 * \param cellRawIds are the raw ids of the cells
 * \param cellValues are the cell values
 * \param problemType is the problem type
 * \param order is the reconstruction order
 * \param[out] interfaceValues are the interface values
 */
__global__ void dev_evalInterfaceBCs(std::size_t nInterfaces, const std::size_t *interfaceRawIds, const int *interfaceBCs,
		                             const double * const *interfaceNormals, const double * const *interfaceCentroids,
                                     const std::size_t *cellRawIds, const double * const *cellValues,
                                     int problemType, int order, double **interfaceValues)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const int interfaceBC = interfaceBCs[i];

    const std::size_t interfaceRawId = interfaceRawIds[i];
    const double interfaceCentroid[3] = {interfaceCentroids[0][interfaceRawId], interfaceCentroids[1][interfaceRawId], interfaceCentroids[2][interfaceRawId]};
    const double interfaceNormal[3]   = {interfaceNormals[0][interfaceRawId], interfaceNormals[1][interfaceRawId], interfaceNormals[2][interfaceRawId]};

    // Cell information
    const std::size_t cellRawId = cellRawIds[i];
    DeviceCollectionDataConstCursor<double> innerValues(cellValues, cellRawId);

    // Evaluate boundary values
    DeviceCollectionDataCursor<double> boundaryValues(interfaceValues, i);
    dev_evalInterfaceBCValues(problemType, interfaceBC, interfaceCentroid, interfaceNormal, innerValues, &boundaryValues);
}

/*!
 * Update residual of cells associated with uniform interfaces.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param interfaceAreas are the areas of the interfaces
 * \param leftCellRawIds are the raw ids of the left cells
 * \param rightCellRawIds are the raw ids of the right cells
 * \param leftReconstructions are the left reconstructions
 * \param rightReconstructions are the right reconstructions
 * \param[out] cellRHS are the RHS of the cells
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
__global__ void dev_uniformUpdateRHS(std::size_t nInterfaces, const std::size_t *interfaceRawIds,
                                     const double * const *interfaceNormals, const double *interfaceAreas,
                                     const std::size_t *leftCellRawIds, const std::size_t *rightCellRawIds,
                                     const double * const *leftReconstructions, const double * const *rightReconstructions,
                                     double **cellRHS, double *maxEig)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];
    const double interfaceNormal[3] = {interfaceNormals[0][interfaceRawId], interfaceNormals[1][interfaceRawId], interfaceNormals[2][interfaceRawId]};
    const double interfaceArea = interfaceAreas[interfaceRawId];

    // Evaluate the conservative fluxes
    DeviceCollectionDataConstCursor<double> leftReconstruction(leftReconstructions, i);
    DeviceCollectionDataConstCursor<double> rightReconstruction(rightReconstructions, i);

    double interfaceFluxes[N_FIELDS];
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_evalSplitting(leftReconstruction, rightReconstruction, interfaceNormal, interfaceFluxes, &interfaceMaxEig);

    // Update cell residuals
    std::size_t leftCellRawId  = leftCellRawIds[i];
    std::size_t rightCellRawId = rightCellRawIds[i];

    DeviceCollectionDataCursor<double> leftRHS(cellRHS, leftCellRawId);
    DeviceCollectionDataCursor<double> rightRHS(cellRHS, rightCellRawId);

    double interfaceCoefficient = interfaceArea;
    for (int k = 0; k < N_FIELDS; ++k) {
        double interfaceContribution = interfaceCoefficient * interfaceFluxes[k];

        atomicAdd(leftRHS.data(k),  - interfaceContribution);
        atomicAdd(rightRHS.data(k),   interfaceContribution);
    }

    // Update maximum eigenvalue
    dev_reduceMax(interfaceMaxEig, nInterfaces, maxEig);
}

/*!
 * Update residual of cells associated with boundary interfaces.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param interfaceAreas are the areas of the interfaces
 * \param fluidCellRawIds are the raw ids of the fluid cells
 * \param boundarySigns are the signs of the boundaries
 * \param fluidReconstructions are the fluid side reconstructions
 * \param virtualReconstructions are the virtual side reconstructions
 * \param[out] cellRHS are the RHS of the cells
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
__global__ void dev_boundaryUpdateRHS(std::size_t nInterfaces, const std::size_t *interfaceRawIds,
                                      const double * const *interfaceNormals, const double *interfaceAreas,
                                      const std::size_t *fluidCellRawIds, const int *boundarySigns,
                                      const double * const *fluidReconstructions, const double * const *virtualReconstructions,
                                      double **cellRHS, double *maxEig)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];
    const double interfaceNormal[3] = {interfaceNormals[0][interfaceRawId], interfaceNormals[1][interfaceRawId], interfaceNormals[2][interfaceRawId]};
    const double interfaceArea = interfaceAreas[interfaceRawId];

    // Info abount the bounday
    const int boundarySign = boundarySigns[i];

    // Evaluate the conservative fluxes
    DeviceCollectionDataConstCursor<double> fluidReconstruction(fluidReconstructions, i);
    DeviceCollectionDataConstCursor<double> virtualReconstruction(virtualReconstructions, i);

    double interfaceFluxes[N_FIELDS];
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_evalSplitting(fluidReconstruction, virtualReconstruction, interfaceNormal, interfaceFluxes, &interfaceMaxEig);

    // Update residual of fluid cell
    std::size_t fluidCellRawId = fluidCellRawIds[i];

    DeviceCollectionDataCursor<double> fluidCellRHS(cellRHS, fluidCellRawId);

    double interfaceCoefficient = boundarySign * interfaceArea;
    for (int k = 0; k < N_FIELDS; ++k) {
        double interfaceContribution = interfaceCoefficient * interfaceFluxes[k];

        atomicAdd(&(fluidCellRHS[k]), - interfaceContribution);
    }

    // Update maximum eigenvalue
    dev_reduceMax(interfaceMaxEig, nInterfaces, maxEig);
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

    double **devLeftReconstructions  = computationInfo.getSolvedInterfaceLeftReconstructions().cuda_deviceCollectionData();
    double **devRightReconstructions = computationInfo.getSolvedInterfaceRightReconstructions().cuda_deviceCollectionData();

    int devProblemType = static_cast<int>(problemType);

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
    const int UNIFORM_BLOCK_SIZE = 256;
    int nUniformBlocks = (nSolvedUniformInterfaces + UNIFORM_BLOCK_SIZE - 1) / UNIFORM_BLOCK_SIZE;
    int uniformSharedMemorySize = UNIFORM_BLOCK_SIZE * sizeof(double);

    // Evaluate interface values
    dev_evalInterfaceValues<<<nUniformBlocks, UNIFORM_BLOCK_SIZE, uniformSharedMemorySize>>>(nSolvedUniformInterfaces, devUniformInterfaceRawIds, devInterfaceCentroids,
                                                                                             devUniformOwnerRawIds, devCellConservatives, order, devLeftReconstructions);

    dev_evalInterfaceValues<<<nUniformBlocks, UNIFORM_BLOCK_SIZE, uniformSharedMemorySize>>>(nSolvedUniformInterfaces, devUniformInterfaceRawIds, devInterfaceCentroids,
                                                                                             devUniformNeighRawIds, devCellConservatives, order, devRightReconstructions);

    // Evaluate fluxes
    dev_uniformUpdateRHS<<<nUniformBlocks, UNIFORM_BLOCK_SIZE, uniformSharedMemorySize>>>(nSolvedUniformInterfaces, devUniformInterfaceRawIds,
                                                                                          devInterfaceNormals, devInterfaceAreas,
                                                                                          devUniformOwnerRawIds, devUniformNeighRawIds,
                                                                                          devLeftReconstructions, devRightReconstructions,
                                                                                          devCellsRHS, devMaxEig);

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
    const int BOUNDARY_BLOCK_SIZE = 256;
    int nBoundaryBlocks = (nBoundaryInterfaces + BOUNDARY_BLOCK_SIZE - 1) / BOUNDARY_BLOCK_SIZE;
    int boundarySharedMemorySize = BOUNDARY_BLOCK_SIZE * sizeof(double);

    // Evaluate interface values
    dev_evalInterfaceValues<<<nBoundaryBlocks, BOUNDARY_BLOCK_SIZE, boundarySharedMemorySize>>>(nBoundaryInterfaces, devBoundaryInterfaceRawIds, devInterfaceCentroids,
                                                                                                devBoundaryFluidRawIds, devCellConservatives,  order, devLeftReconstructions);

    dev_evalInterfaceBCs<<<nBoundaryBlocks, BOUNDARY_BLOCK_SIZE, boundarySharedMemorySize>>>(nBoundaryInterfaces, devBoundaryInterfaceRawIds, devBoundaryInterfaceBCs,
                                                                                             devInterfaceNormals, devInterfaceCentroids,
                                                                                             devBoundaryFluidRawIds, devCellConservatives,
                                                                                             devProblemType, order, devRightReconstructions);

    // Evaluate fluxes
    dev_boundaryUpdateRHS<<<nBoundaryBlocks, BOUNDARY_BLOCK_SIZE, boundarySharedMemorySize>>>(nBoundaryInterfaces, devBoundaryInterfaceRawIds,
                                                                                              devInterfaceNormals, devInterfaceAreas,
                                                                                              devBoundaryFluidRawIds, devBoundaryInterfaceSigns,
                                                                                              devLeftReconstructions, devRightReconstructions,
                                                                                              devCellsRHS, devMaxEig);

    //
    // Update host memory
    //
    cellsRHS->cuda_updateHost();
    CUDA_ERROR_CHECK(cudaMemcpy(maxEig, devMaxEig, sizeof(double), cudaMemcpyDeviceToHost));
}

}
