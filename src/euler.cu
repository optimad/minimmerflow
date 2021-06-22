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
#include "reconstruction.hpp"
#include "utils_cuda.hpp"

#define uint64  unsigned long long

namespace euler {

/**
 * @brief Compute the maximum of 2 double-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
__device__ double atomicMax(double *const address, const double value)
{
    uint64 *address_as_i =(uint64*)address;
    uint64 old = *address_as_i, assumed;
    while (value > __longlong_as_double(old)) {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __double_as_longlong(value));
        }
    return __longlong_as_double(old);
}


/**
 * @brief Compute the maximum of double-precision floating point values
 *
 * @param[in]	d_array  The address of the value which is the candidate for the  the max. eigenvalue
 * @param[in]	d_max    The address of the max. value
 * @param[in]	elements The number of elements which are parsed
 * @param[in]	value	 The address of the shared memory values
 */
__device__ void max_reduce(const double* const d_array, double* d_max, const size_t elements, double *shared)
{
    int tid = threadIdx.x;
    shared[tid] = *d_array;
    __syncthreads();
    int gid = (blockDim.x * blockIdx.x) + tid;
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s && gid < elements)
            shared[tid] = max(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
      atomicMax(d_max, shared[0]);
}


/*!
 * Calculates the conservative fluxes for a perfect gas.
 *
 * \param conservative is the conservative state
 * \param primitive is the primitive state
 * \param n is the normal direction
 * \param[out] fluxes on output will contain the conservative fluxes
 */
__device__ void dev_evalFluxes(const double *conservative, const double *primitive, const double *n, double *fluxes)
{
    // Compute variables
    double u = primitive[DEV_FID_U];
    double v = primitive[DEV_FID_V];
    double w = primitive[DEV_FID_W];

    double vel2 = u * u + v * v + w * w;
    double un   = ::utils::dev_normalVelocity(primitive, n);

    double p = primitive[DEV_FID_P];
    if (p < 0.) {
        printf("***** Negative pressure (%f) in flux computation!\n", p);
    }

    double rho = conservative[DEV_FID_RHO];
    if (rho < 0.) {
       printf("***** Negative density (%f) in flux computation!\n", rho);
    }

    double eto = p / (DEV_GAMMA - 1.) + 0.5 * rho * vel2;

    // Compute fluxes
    double massFlux = rho * un;

    fluxes[DEV_FID_EQ_C]   = massFlux;
    fluxes[DEV_FID_EQ_M_X] = massFlux * u + p * n[0];
    fluxes[DEV_FID_EQ_M_Y] = massFlux * v + p * n[1];
    fluxes[DEV_FID_EQ_M_Z] = massFlux * w + p * n[2];
    fluxes[DEV_FID_EQ_E]   = un * (eto + p);
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
__device__ void dev_evalSplitting(const double *conservativeL, const double *conservativeR, const double *n, double *fluxes, double *lambda)
{
    // Primitive variables
    double primitiveL[N_FIELDS];
    ::utils::dev_conservative2primitive(conservativeL, primitiveL);

    double primitiveR[N_FIELDS];
    ::utils::dev_conservative2primitive(conservativeR, primitiveR);

    // Fluxes
    double fL[N_FIELDS];
    dev_evalFluxes(conservativeL, primitiveL, n, fL);

    double fR[N_FIELDS];
    dev_evalFluxes(conservativeR, primitiveR, n, fR);

    // Eigenvalues
    double unL     = ::utils::dev_normalVelocity(primitiveL, n);
    double aL      = std::sqrt(GAMMA * primitiveL[DEV_FID_T]);
    double lambdaL = std::abs(unL) + aL;

    double unR     = ::utils::dev_normalVelocity(primitiveR, n);
    double aR      = std::sqrt(DEV_GAMMA * primitiveR[DEV_FID_T]);
    double lambdaR = std::abs(unR) + aR;

    *lambda = max(lambdaR, lambdaL);

    // Splitting
    for (int k = 0; k < N_FIELDS; ++k) {
        fluxes[k] = 0.5 * ((fR[k] + fL[k]) - (*lambda) * (conservativeR[k] - conservativeL[k]));
    }
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
                                     const double *interfaceNormals, const double *interfaceAreas,
                                     const std::size_t *leftCellRawIds, const std::size_t *rightCellRawIds,
                                     const double *leftReconstructions, const double *rightReconstructions,
                                     double *cellRHS, double *maxEig)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];

    // Info about the interface
    const double *interfaceNormal = interfaceNormals + 3 * interfaceRawId;
    const double interfaceArea    = interfaceAreas[interfaceRawId];

    // Evaluate the conservative fluxes
    const double *leftReconstruction  = leftReconstructions  + N_FIELDS * i;
    const double *rightReconstruction = rightReconstructions + N_FIELDS * i;

    double interfaceFluxes[N_FIELDS];
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_evalSplitting(leftReconstruction, rightReconstruction, interfaceNormal, interfaceFluxes, &interfaceMaxEig);

    // Update residual of left cell
    std::size_t leftCellRawId = leftCellRawIds[i];
    double *leftRHS = cellRHS + N_FIELDS * leftCellRawId;
    for (int k = 0; k < N_FIELDS; ++k) {
        atomicAdd(leftRHS + k, - interfaceArea * interfaceFluxes[k]);
    }

    // Update residual of right cell
    std::size_t rightCellRawId = rightCellRawIds[i];
    double *rightRHS = cellRHS + N_FIELDS * rightCellRawId;
    for (int k = 0; k < N_FIELDS; ++k) {
        atomicAdd(rightRHS + k, interfaceArea * interfaceFluxes[k]);
    }

    // Update maximum eigenvalue
    extern __shared__ double shared[];
    max_reduce(&interfaceMaxEig, maxEig, nInterfaces, shared);
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
                                      const double *interfaceNormals, const double *interfaceAreas,
                                      const std::size_t *fluidCellRawIds, const std::size_t *boundarySigns,
                                      const double *fluidReconstructions, const double *virtualReconstructions,
                                      double *cellRHS, double *maxEig)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];

    // Info about the interface
    const double *interfaceNormal = interfaceNormals + 3 * interfaceRawId;
    const double interfaceArea    = interfaceAreas[interfaceRawId];

    // Info abount the bounday
    const int boundarySign = boundarySigns[i];

    // Evaluate the conservative fluxes
    const double *fluidReconstruction   = fluidReconstructions   + N_FIELDS * i;
    const double *virtualReconstruction = virtualReconstructions + N_FIELDS * i;

    double interfaceFluxes[N_FIELDS];
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_evalSplitting(fluidReconstruction, virtualReconstruction, interfaceNormal, interfaceFluxes, &interfaceMaxEig);

    // Update residual of fluid cell
    std::size_t fluidCellRawId = fluidCellRawIds[i];
    double *fluidRHS = cellRHS + N_FIELDS * fluidCellRawId;
    for (int k = 0; k < N_FIELDS; ++k) {
        atomicAdd(fluidRHS + k, - boundarySign * interfaceArea * interfaceFluxes[k]);
    }

    // Update maximum eigenvalue
    atomicMax(maxEig, interfaceMaxEig);
}

/*!
 * Reset the RHS.
 *
 * \param[in,out] rhs is the RHS that will be reset
 */
void cuda_resetRHS(ScalarPiercedStorage<double> *cellsRHS)
{
    cellsRHS->cuda_fillDevice(0.);
    cellsRHS->cuda_updateHost();
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
                    const ScalarPiercedStorage<double> &cellConservatives, ScalarPiercedStorage<double> *cellsRHS, double *maxEig)
{
    //
    // Initialization
    //
    const double *devInterfaceNormals = computationInfo.cuda_getInterfaceNormalDevData();
    const double *devInterfaceAreas   = computationInfo.cuda_getInterfaceAreaDevData();

    double *devMaxEig;
    CUDA_ERROR_CHECK(cudaMalloc((void **) &devMaxEig, 1 * sizeof(double)));
    CUDA_ERROR_CHECK(cudaMemset(devMaxEig, 0., 1 * sizeof(double)));

    double *devCellsRHS = cellsRHS->cuda_deviceData();

    const ScalarStorage<std::size_t> &solvedUniformInterfaceRawIds = computationInfo.getSolvedUniformInterfaceRawIds();
    const std::size_t nSolvedUniformInterfaces = solvedUniformInterfaceRawIds.size();

    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceRawIds = computationInfo.getSolvedBoundaryInterfaceRawIds();
    const std::size_t nSolvedBoundaryInterfaces = solvedBoundaryInterfaceRawIds.size();

    ScalarStorage<double> *leftReconstructions = &(computationInfo.getSolvedInterfaceLeftReconstructions());
    ScalarStorage<double> *rightReconstructions = &(computationInfo.getSolvedInterfaceRightReconstructions());

    double *devLeftReconstructions = computationInfo.getSolvedInterfaceLeftReconstructions().cuda_deviceData();
    double *devRightReconstructions = computationInfo.getSolvedInterfaceRightReconstructions().cuda_deviceData();

    //
    // Initialize residual
    //

    // Reset the residuals
    cuda_resetRHS(cellsRHS);

    //
    // Process uniform interfaces
    //
    const ScalarStorage<std::size_t> &solvedUniformInterfaceOwnerRawIds = computationInfo.getSolvedUniformInterfaceOwnerRawIds();
    const ScalarStorage<std::size_t> &solvedUniformInterfaceNeighRawIds = computationInfo.getSolvedUniformInterfaceNeighRawIds();

    // Reconstruct interface values
    for (std::size_t i = 0; i < nSolvedUniformInterfaces; ++i) {
        // Info about the interface
        const std::size_t interfaceRawId = solvedUniformInterfaceRawIds[i];
        const std::array<double, 3> &interfaceCentroid = computationInfo.rawGetInterfaceCentroid(interfaceRawId);

        // Info about the interface owner
        std::size_t ownerRawId = solvedUniformInterfaceOwnerRawIds[i];
        const double *ownerMean = cellConservatives.rawData(ownerRawId);

        // Info about the interface neighbour
        std::size_t neighRawId = solvedUniformInterfaceNeighRawIds[i];
        const double *neighMean = cellConservatives.rawData(neighRawId);

        // Evaluate interface reconstructions
        double *ownerReconstruction = leftReconstructions->data() + N_FIELDS * i;
        double *neighReconstruction = rightReconstructions->data() + N_FIELDS * i;

        reconstruction::eval(ownerRawId, computationInfo, order, interfaceCentroid, ownerMean, ownerReconstruction);
        reconstruction::eval(neighRawId, computationInfo, order, interfaceCentroid, neighMean, neighReconstruction);
    }

    leftReconstructions->cuda_updateDevice(N_FIELDS * nSolvedUniformInterfaces);
    rightReconstructions->cuda_updateDevice(N_FIELDS * nSolvedUniformInterfaces);

    // Evaluate fluxes
    const std::size_t *devSolvedUniformInterfaceRawIds = solvedUniformInterfaceRawIds.cuda_deviceData();

    const std::size_t *devUniformOwnerRawIds = solvedUniformInterfaceOwnerRawIds.cuda_deviceData();
    const std::size_t *devUniformNeighRawIds = solvedUniformInterfaceNeighRawIds.cuda_deviceData();

    const int UNIFORM_BLOCK_SIZE = 256;
    int nUniformnBlocks = (nSolvedUniformInterfaces + UNIFORM_BLOCK_SIZE - 1) / UNIFORM_BLOCK_SIZE;
    dev_uniformUpdateRHS<<<nUniformnBlocks, UNIFORM_BLOCK_SIZE, UNIFORM_BLOCK_SIZE*sizeof(double)>>>(nSolvedUniformInterfaces, devSolvedUniformInterfaceRawIds,
                                                                                                     devInterfaceNormals, devInterfaceAreas,
                                                                                                     devUniformOwnerRawIds, devUniformNeighRawIds,
                                                                                                     devLeftReconstructions, devRightReconstructions,
                                                                                                     devCellsRHS, devMaxEig);

    double uniformMaxEig;
    CUDA_ERROR_CHECK(cudaMemcpy(&uniformMaxEig, devMaxEig, 1 * sizeof(double), cudaMemcpyDeviceToHost));

    //
    // Process boundary interfaces
    //
    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceSigns = computationInfo.getSolvedBoundaryInterfaceSigns();
    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceFluidRawIds = computationInfo.getSolvedBoundaryInterfaceFluidRawIds();

    // Reconstruct interface values
    for (std::size_t i = 0; i < nSolvedBoundaryInterfaces; ++i) {
        // Info about the interface
        const std::size_t interfaceRawId = solvedBoundaryInterfaceRawIds[i];
        const std::array<double, 3> &interfaceNormal = computationInfo.rawGetInterfaceNormal(interfaceRawId);
        const std::array<double, 3> &interfaceCentroid = computationInfo.rawGetInterfaceCentroid(interfaceRawId);
        int interfaceBC = solvedBoundaryInterfaceBCs[i];

        // Info about the interface fluid cell
        std::size_t fluidRawId = solvedBoundaryInterfaceFluidRawIds[i];
        const double *fluidMean = cellConservatives.rawData(fluidRawId);

        // Evaluate interface reconstructions
        double *fluidReconstruction   = leftReconstructions->data() + N_FIELDS * i;
        double *virtualReconstruction = rightReconstructions->data() + N_FIELDS * i;

        reconstruction::eval(fluidRawId, computationInfo, order, interfaceCentroid, fluidMean, fluidReconstruction);
        evalInterfaceBCValues(problemType, interfaceBC, interfaceCentroid, interfaceNormal, fluidReconstruction, virtualReconstruction);
    }

    leftReconstructions->cuda_updateDevice(N_FIELDS * nSolvedBoundaryInterfaces);
    rightReconstructions->cuda_updateDevice(N_FIELDS * nSolvedBoundaryInterfaces);

    // Evaluate fluxes
    const std::size_t *devSolvedBoundaryInterfaceRawIds = solvedBoundaryInterfaceRawIds.cuda_deviceData();

    const std::size_t *devBoundaryFluidRawIds = solvedBoundaryInterfaceFluidRawIds.cuda_deviceData();

    const std::size_t *devBoundarySigns = solvedBoundaryInterfaceSigns.cuda_deviceData();

    const int BOUNDARY_BLOCK_SIZE = 256;
    int nBoundarynBlocks = (nSolvedBoundaryInterfaces + BOUNDARY_BLOCK_SIZE - 1) / BOUNDARY_BLOCK_SIZE;
    dev_boundaryUpdateRHS<<<nBoundarynBlocks, UNIFORM_BLOCK_SIZE>>>(nSolvedBoundaryInterfaces, devSolvedBoundaryInterfaceRawIds,
                                                                    devInterfaceNormals, devInterfaceAreas,
                                                                    devBoundaryFluidRawIds, devBoundarySigns,
                                                                    devLeftReconstructions, devRightReconstructions,
                                                                    devCellsRHS, devMaxEig);

    double boundaryMaxEig;
    CUDA_ERROR_CHECK(cudaMemcpy(&boundaryMaxEig, devMaxEig, 1 * sizeof(double), cudaMemcpyDeviceToHost));

    // Evaluate maximum eigenvalue
    *maxEig = std::max(uniformMaxEig, boundaryMaxEig);

    //
    // Update host memory
    //

    cellsRHS->cuda_updateHost();

    //
    // Clean-up
    //
    CUDA_ERROR_CHECK(cudaFree(devMaxEig));
}

}
