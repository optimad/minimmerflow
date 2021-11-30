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
    workspace[tid] = value;
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
 * Calculates the conservative fluxes for a perfect gas.
 *
 * \param conservative is the conservative state
 * \param n is the normal direction
 * \param[out] fluxes on output will contain the conservative fluxes
 * \param[out] lambda on output will contain the maximum eigenvalue
 */
__device__ void dev_evalFluxes(const DeviceSharedArray<double> &conservative, const double *normal,
                               DeviceSharedArray<double> *fluxes, double *lambda)
{
    // Compute primitive variables
    //
    // To reduce the usage of shared memory, primitive fields are evaluated on
    // the storage that will hold the fluxes.
    DeviceSharedArray<double> &primitive = *fluxes;

    ::utils::dev_conservative2primitive(conservative, &primitive);

    // Speed of sound
    double T = primitive[DEV_FID_T];
    double a = std::sqrt(DEV_GAMMA * T);

    // Compute normal velocity
    double nx = normal[0];
    double ny = normal[1];
    double nz = normal[2];

    double u = primitive[DEV_FID_U];
    double v = primitive[DEV_FID_V];
    double w = primitive[DEV_FID_W];

    double un = ::utils::dev_normalVelocity(u, v, w, nx, ny, nz);

    // Evaluate maximum eigenvalue
    *lambda = std::abs(un) + a;

    // Mass flux
    double rho = conservative[DEV_FID_RHO];
    if (rho < 0.) {
       printf("***** Negative density (%f) in flux computation!\n", rho);
    }

    double massFlux = rho * un;

    // Energy flux
    double p = primitive[DEV_FID_P];
    if (p < 0.) {
        printf("***** Negative pressure (%f) in flux computation!\n", p);
    }

    double rho_K = 0.5 * rho * (u * u + v * v + w * w);
    double eto   = p / (DEV_GAMMA - 1.) + rho_K;

    (*fluxes)[DEV_FID_EQ_E] = un * (eto + p);

    // Momentum flux
    (*fluxes)[DEV_FID_EQ_M_X] = massFlux * u + p * nx;
    (*fluxes)[DEV_FID_EQ_M_Y] = massFlux * v + p * ny;
    (*fluxes)[DEV_FID_EQ_M_Z] = massFlux * w + p * nz;

    // Continuity flux
    (*fluxes)[DEV_FID_EQ_C] = massFlux;
}

/*!
 * Solve the given Riemann problem using the Local Lax Friedrichs approximate
 * solver.
 *
 * \param conservativeL is the left conservative state
 * \param conservativeR is the right conservative state
 * \param n is the normal
 * \param[out] fluxes on output will contain the conservative fluxes
 * \param[out] lambda on output will contain the maximum eigenvalue
 */
__device__ void dev_solveRiemann(const DeviceSharedArray<double> &conservativeL, const DeviceSharedArray<double> &conservativeR,
                                 const double *normal, DeviceSharedArray<double> *fluxes, double *lambda)
{
    // Fluxes on the left side
    //
    // To reduce the usage of shared memory, fluxes are temporary evaluated on
    // the storage that will hold the interface fluxes and then copied on a
    // local array.
    dev_evalFluxes(conservativeL, normal, fluxes, lambda);

    double fL[N_FIELDS];
    for (int k = 0; k < N_FIELDS; ++k) {
        fL[k] = (*fluxes)[k];
    }

    // Fluxes on the right side
    //
    // To reduce the usage of shared memory, fluxes are evaluated on the
    // storage that will hold the interface fluxes.
    double lambdaR;
    dev_evalFluxes(conservativeR, normal, fluxes, &lambdaR);
    *lambda = max(lambdaR, *lambda);

    DeviceSharedArray<double> &fR = *fluxes;

    // Splitting
    for (int k = 0; k < N_FIELDS; ++k) {
        (*fluxes)[k] = 0.5 * ((fR[k] + fL[k]) - (*lambda) * (conservativeR[k] - conservativeL[k]));
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
__device__ void dev_evalInterfaceValues(int order, const double *point,
                                        const DeviceCollectionDataConstCursor<double> &means,
                                        DeviceSharedArray<double> *values)
{
    reconstruction::dev_eval(order, point, means, values);
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
__device__ void dev_evalInterfaceBCValues(int problemType, int BCType, const double *point, const double *normal,
                                          const DeviceSharedArray<double> &innerValues, DeviceSharedArray<double> *boundaryValues)
{
    double infoStorage[BC_INFO_SIZE];
    DeviceProxyArray<double> info(&(infoStorage[0]), 0, 1);
    problem::dev_getBorderBCInfo(problemType, BCType, point, normal, &info);

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
 * Computes the boundary values for the free flow BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner innerValues values
 * \param[out] boundaryValues are the boundary values
 */
__device__ void dev_evalFreeFlowBCValues(const double *point, const double *normal, const DeviceProxyArray<double> &info,
                                         const DeviceSharedArray<double> &innerValues, DeviceSharedArray<double> *boundaryValues)
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
__device__ void dev_evalReflectingBCValues(const double *point, const double *normal, const DeviceProxyArray<double> &info,
                                           const DeviceSharedArray<double> &innerValues, DeviceSharedArray<double> *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(info);

    // Evaluate primitive variables
    //
    // To reduce the usage of shared memory, primitive variables are temporary
    // evaluated on the storage that will hold boundary values and then copied
    // on a local array.
    ::utils::dev_conservative2primitive(innerValues, boundaryValues);

    double primitive[N_FIELDS];
    for (int k = 0; k < N_FIELDS; ++k) {
        primitive[k] = (*boundaryValues)[k];
    }

    // Apply boundary condition
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

    // Evaluate conservative values
    DeviceProxyArray<double> immutablePrimitive(&(primitive[0]), 0, 1);
    ::utils::dev_primitive2conservative(immutablePrimitive, boundaryValues);
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
__device__ void dev_evalWallBCValues(const double *point, const double *normal, const DeviceProxyArray<double> &info,
                                     const DeviceSharedArray<double> &innerValues, DeviceSharedArray<double> *boundaryValues)
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
__device__ void dev_evalDirichletBCValues(const double *point, const double *normal, const DeviceProxyArray<double> &info,
                                          const DeviceSharedArray<double> &innerValues, DeviceSharedArray<double> *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);
    BITPIT_UNUSED(innerValues);

    ::utils::dev_primitive2conservative(info, boundaryValues);
}

/*!
 * Update residual of cells associated with uniform interfaces.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param reconstructionOrder is the order at wich values will be reconstructed
 * on the interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param interfaceAreas are the areas of the interfaces
 * \param interfaceCentroids are the centroids of the interfaces
 * \param leftCellRawIds are the raw ids of the left cells
 * \param rightCellRawIds are the raw ids of the right cells
 * \param cellConvervatives are the conservative fileds on the cells
 * \param[out] cellRHS are the RHS of the cells
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
__global__ void dev_uniformUpdateRHS(std::size_t nInterfaces, int reconstructionOrder,
                                     const std::size_t *interfaceRawIds, const double *interfaceAreas,
                                     const double * const *interfaceNormals, const double * const *interfaceCentroids,
                                     const std::size_t *leftCellRawIds, const std::size_t *rightCellRawIds,
                                     const double * const *cellConvervatives, double **cellRHS, double *maxEig)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    // Initialize shared storage
    //
    // The shared storage is logically divided in three slots, each of these
    // slots can contain a sets of conservative variables.
    extern __shared__ double sharedStorage[];

    const std::size_t sharedSlotSize = DeviceSharedArray<double>::evaluateSharedSize(N_FIELDS);

    // Get interface information
    const std::size_t interfaceRawId = interfaceRawIds[i];

    // Cell information
    const std::size_t leftCellRawId  = leftCellRawIds[i];
    const std::size_t rightCellRawId = rightCellRawIds[i];

    //
    // Reconstruct interface values
    //
    // During this stage usage of shared memory is the following:
    //    Slot #1: not used
    //    Slot #2: left side reconstructed variables
    //    Slot #3: right side reconstructed variables
    const double interfaceCentroid[3] = {interfaceCentroids[0][interfaceRawId], interfaceCentroids[1][interfaceRawId], interfaceCentroids[2][interfaceRawId]};

    DeviceCollectionDataConstCursor<double> cellConservativesCursor(cellConvervatives, 0);

    cellConservativesCursor.set(leftCellRawId);
    double *leftReconstructionsStorage = &(sharedStorage[sharedSlotSize]);
    DeviceSharedArray<double> leftInterfaceConservatives(leftReconstructionsStorage);
    dev_evalInterfaceValues(reconstructionOrder, interfaceCentroid, cellConservativesCursor, &leftInterfaceConservatives);

    cellConservativesCursor.set(rightCellRawId);
    double *rightReconstructionsStorage = &(sharedStorage[2 * sharedSlotSize]);
    DeviceSharedArray<double> rightInterfaceConservatives(rightReconstructionsStorage);
    dev_evalInterfaceValues(reconstructionOrder, interfaceCentroid, cellConservativesCursor, &rightInterfaceConservatives);

    //
    // Evaluate interface fluxes
    //
    // During this stage usage of shared memory is the following:
    //    Slot #1: interface fluxes
    //    Slot #2: left side reconstructed variables
    //    Slot #3: right side reconstructed variables
    const double interfaceNormal[3] = {interfaceNormals[0][interfaceRawId], interfaceNormals[1][interfaceRawId], interfaceNormals[2][interfaceRawId]};

    double *interfaceFluxesStorage = &(sharedStorage[0]);
    DeviceSharedArray<double> interfaceFluxes(interfaceFluxesStorage);
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_solveRiemann(leftInterfaceConservatives, rightInterfaceConservatives, interfaceNormal, &interfaceFluxes, &interfaceMaxEig);

    //
    // Update cell residuals
    //
    // During this stage usage of shared memory is the following:
    //    Slot #1: interface fluxes
    //    Slot #2: not used
    //    Slot #3: not used
    const double interfaceCoefficient = interfaceAreas[interfaceRawId];
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] *= interfaceCoefficient;
    }

    DeviceCollectionDataCursor<double> residuals(cellRHS, 0);

    residuals.set(leftCellRawId);
    for (int k = 0; k < N_FIELDS; ++k) {
        atomicAdd(residuals.data(k), - interfaceFluxes[k]);
    }

    residuals.set(rightCellRawId);
    for (int k = 0; k < N_FIELDS; ++k) {
        atomicAdd(residuals.data(k), interfaceFluxes[k]);
    }

    // Update maximum eigenvalue
    //
    // During this stage usage of shared memory is the following:
    //    Slot #1: workspace for eigenvalue reduction
    //    Slot #2: not used
    //    Slot #3: not used
    double *reductionWorkspace = &(sharedStorage[0]);
    dev_reduceMax(interfaceMaxEig, nInterfaces, reductionWorkspace, maxEig);
}

/*!
 * Update residual of cells associated with boundary interfaces.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceAreas are the areas of the interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param interfaceCentroids are the centroids of the interfaces
 * \param fluidCellRawIds are the raw ids of the fluid cells
 * \param cellConvervatives are the conservative fileds on the cells
 * \param boundarySigns are the signs of the boundaries
 * \param[out] cellRHS are the RHS of the cells
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
__global__ void dev_boundaryUpdateRHS(std::size_t nInterfaces, int problemType, int reconstructionOrder,
                                      const std::size_t *interfaceRawIds, const double *interfaceAreas,
                                      const double * const *interfaceNormals, const double * const *interfaceCentroids,
                                      const std::size_t *fluidCellRawIds, const double * const *cellConvervatives,
                                      const int *boundarySigns, const int *boundaryBCs, double **cellRHS, double *maxEig)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    // Initialize shared storage
    //
    // The shared storage is logically divided in three slots, each of these
    // slots can contain a sets of conservative variables.
    extern __shared__ double sharedStorage[];

    const std::size_t sharedSlotSize = DeviceSharedArray<double>::evaluateSharedSize(N_FIELDS);

    // Get interface information
    const std::size_t interfaceRawId = interfaceRawIds[i];

    // Cell information
    const std::size_t fluidCellRawId = fluidCellRawIds[i];

    //
    // Reconstruct interface values
    //
    // During this stage usage of shared memory is the following:
    //    Slot #1: not used
    //    Slot #2: fluid side reconstructed variables
    //    Slot #3: virtual side reconstructed conservative variables
    const double interfaceCentroid[3] = {interfaceCentroids[0][interfaceRawId], interfaceCentroids[1][interfaceRawId], interfaceCentroids[2][interfaceRawId]};
    const double interfaceNormal[3]   = {interfaceNormals[0][interfaceRawId], interfaceNormals[1][interfaceRawId], interfaceNormals[2][interfaceRawId]};

    DeviceCollectionDataConstCursor<double> fluidCellConservativesCursor(cellConvervatives, fluidCellRawId);
    double *fluidReconstructionStorage = &(sharedStorage[sharedSlotSize]);
    DeviceSharedArray<double> fluidInterfaceConservatives(fluidReconstructionStorage);
    dev_evalInterfaceValues(reconstructionOrder, interfaceCentroid, fluidCellConservativesCursor, &fluidInterfaceConservatives);

    const int interfaceBC = boundaryBCs[i];
    double *virtualReconstructionStorage = &(sharedStorage[2 * sharedSlotSize]);
    DeviceSharedArray<double> virtualInterfaceConservatives(virtualReconstructionStorage);
    dev_evalInterfaceBCValues(problemType, interfaceBC, interfaceCentroid, interfaceNormal, fluidInterfaceConservatives, &virtualInterfaceConservatives);

    //
    // Evaluate interface fluxes
    //
    // During this stage usage of shared memory is the following:
    //    Slot #1: interface fluxes
    //    Slot #2: fluid side reconstructed variables
    //    Slot #3: virtual side reconstructed variables
    double *interfaceFluxesStorage = &(sharedStorage[0]);
    DeviceSharedArray<double> interfaceFluxes(interfaceFluxesStorage);
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_solveRiemann(fluidInterfaceConservatives, virtualInterfaceConservatives, interfaceNormal, &interfaceFluxes, &interfaceMaxEig);

    //
    // Update cell residuals
    //
    // During this stage usage of shared memory is the following:
    //    Slot #1: interface fluxes
    //    Slot #2: not used
    //    Slot #3: not used
    const int boundarySign     = boundarySigns[i];
    const double interfaceArea = interfaceAreas[interfaceRawId];

    const double interfaceCoefficient = boundarySign * interfaceArea;
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] *= interfaceCoefficient;
    }

    DeviceCollectionDataCursor<double> fluidCellRHS(cellRHS, fluidCellRawId);
    for (int k = 0; k < N_FIELDS; ++k) {
        atomicAdd(&(fluidCellRHS[k]), - interfaceFluxes[k]);
    }

    // Update maximum eigenvalue
    //
    // During this stage usage of shared memory is the following:
    //    Slot #1: workspace for eigenvalue reduction
    //    Slot #2: not used
    //    Slot #3: not used
    double *reductionWorkspace = &(sharedStorage[0]);
    dev_reduceMax(interfaceMaxEig, nInterfaces, reductionWorkspace, maxEig);
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
    int nUniformBlocks = (nSolvedUniformInterfaces + UNIFORM_BLOCK_SIZE - 1) / UNIFORM_BLOCK_SIZE;

    // Evaluate fluxes
    dev_uniformUpdateRHS<<<nUniformBlocks, UNIFORM_BLOCK_SIZE, UNIFORM_SHARED_SIZE>>>
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
    int nBoundaryBlocks = (nBoundaryInterfaces + BOUNDARY_BLOCK_SIZE - 1) / BOUNDARY_BLOCK_SIZE;

    // Evaluate fluxes
    dev_boundaryUpdateRHS<<<nBoundaryBlocks, BOUNDARY_BLOCK_SIZE, BOUNDARY_SHARED_SIZE>>>
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
