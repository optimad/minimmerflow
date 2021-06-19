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

#define uint64  unsigned long long

namespace euler {

/**
 * @brief Compute the maximum of 2 double-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
__device__ void atomicMax(double * const address, const double value)
{
    if (* address >= value) {
        return;
    }

    uint64 * const address_as_i = (uint64 *)address;
    uint64 old = * address_as_i;

    uint64 assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= value) {
            break;
        }

        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
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
 * Compute interface fluxes.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param leftReconstructions are the left reconstructions
 * \param rightReconstructions are the right reconstructions
 * \param[out] interfacesFluxes on output will containt the interface fluxes
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
__global__ void dev_computeInterfaceFluxes(std::size_t nInterfaces, const std::size_t *interfaceRawIds, const double *interfaceNormals,
                                           const double *leftReconstructions, const double *rightReconstructions,
                                           double *interfacesFluxes, double *maxEig)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];

    // Info about the interface
    const double *interfaceNormal = interfaceNormals + 3 * interfaceRawId;

    // Evaluate the conservative fluxes
    const double *leftReconstruction  = leftReconstructions  + N_FIELDS * i;
    const double *rightReconstruction = rightReconstructions + N_FIELDS * i;

    double *interfaceFluxes = interfacesFluxes + N_FIELDS * i;
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_evalSplitting(leftReconstruction, rightReconstruction, interfaceNormal, interfaceFluxes, &interfaceMaxEig);

    atomicMax(maxEig, interfaceMaxEig);
}

/*!
 * Compute interface fluxes.
 *
 * \param computationInfo are the computation information
 * \param &interfaceRawIds are the raw ids of the interfaces that will be processed
 * \param leftReconstructions is the storage for the left reconstructions
 * \param rightReconstructions is the storage for the right reconstructions
 * \param[out] interfacesFluxes on output will containt the interface fluxes
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
void cuda_computeInterfaceFluxes(const ComputationInfo &computationInfo, const ScalarStorage<std::size_t> &interfaceRawIds,
                                 const ScalarStorage<double> &leftReconstructions, const ScalarStorage<double> &rightReconstructions,
                                 ScalarStorage<double> *interfacesFluxes, double *maxEig)
{
    const std::size_t nInterfaces = interfaceRawIds.size();

    // Evaluate fluxes
    const std::size_t *devInterfaceRawIds = interfaceRawIds.cuda_devData();

    const double *devInterfaceNormals = computationInfo.cuda_getInterfaceNormalDevData();

    const double *devLeftReconstructions  = leftReconstructions.cuda_devData();
    const double *devRightReconstructions = rightReconstructions.cuda_devData();

    double *devInterfacesFluxes = interfacesFluxes->cuda_devData();

    double *devMaxEig;
    cudaMalloc((void **) &devMaxEig, 1 * sizeof(double));
    cudaMemset(devMaxEig, 0., 1 * sizeof(double));

    int blockSize = 256;
    int numBlocks = (nInterfaces + blockSize - 1) / blockSize;
    dev_computeInterfaceFluxes<<<numBlocks, blockSize>>>(nInterfaces, devInterfaceRawIds, devInterfaceNormals,
                                                         devLeftReconstructions, devRightReconstructions,
                                                         devInterfacesFluxes, devMaxEig);

    // Update host data
    interfacesFluxes->cuda_updateHost();
    cudaMemcpy(maxEig, devMaxEig, 1 * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(devMaxEig);
}

/*!
 * Reset the RHS.
 *
 * \param[in,out] rhs is the RHS that will be reset
 */
void cuda_resetRHS(ScalarPiercedStorage<double> *cellsRHS)
{
    cellsRHS->cuda_devFill(0.);
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
void cuda_updateRHS(problem::ProblemType problemType, const ComputationInfo &computationInfo,
                    const int order, const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
                    const ScalarPiercedStorage<double> &cellConservatives, ScalarPiercedStorage<double> *cellsRHS, double *maxEig)
{
    //
    // Initialize residual
    //

    // Reset the residuals
    cuda_resetRHS(cellsRHS);

    //
    // Process uniform interfaces
    //
    const ScalarStorage<std::size_t> &solvedUniformInterfaceRawIds = computationInfo.getSolvedUniformInterfaceRawIds();
    const ScalarStorage<std::size_t> &solvedUniformInterfaceOwnerRawIds = computationInfo.getSolvedUniformInterfaceOwnerRawIds();
    const ScalarStorage<std::size_t> &solvedUniformInterfaceNeighRawIds = computationInfo.getSolvedUniformInterfaceNeighRawIds();
    const std::size_t nSolvedUniformInterfaces = solvedUniformInterfaceRawIds.size();

    // Evaluate interface reconstructions
    ScalarStorage<double> uniformOwnerReconstructions(N_FIELDS * nSolvedUniformInterfaces);
    ScalarStorage<double> uniformNeighReconstructions(N_FIELDS * nSolvedUniformInterfaces);
    uniformOwnerReconstructions.cuda_allocate();
    uniformNeighReconstructions.cuda_allocate();

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
        double *ownerReconstruction = uniformOwnerReconstructions.data() + N_FIELDS * i;
        double *neighReconstruction = uniformNeighReconstructions.data() + N_FIELDS * i;

        reconstruction::eval(ownerRawId, computationInfo, order, interfaceCentroid, ownerMean, ownerReconstruction);
        reconstruction::eval(neighRawId, computationInfo, order, interfaceCentroid, neighMean, neighReconstruction);
    }

    uniformOwnerReconstructions.cuda_updateDevice();
    uniformNeighReconstructions.cuda_updateDevice();

    // Evaluate fluxes
    ScalarStorage<double> uniformInterfacesFluxes(N_FIELDS * nSolvedUniformInterfaces);
    uniformInterfacesFluxes.cuda_allocate();

    double uniformMaxEig;
    cuda_computeInterfaceFluxes(computationInfo, solvedUniformInterfaceRawIds, uniformOwnerReconstructions, uniformNeighReconstructions, &uniformInterfacesFluxes, &uniformMaxEig);

    // Update the residuals
    for (std::size_t i = 0; i < nSolvedUniformInterfaces; ++i) {
        // Info about the interface
        const std::size_t interfaceRawId = solvedUniformInterfaceRawIds[i];
        const double interfaceArea = computationInfo.rawGetInterfaceArea(interfaceRawId);
        const double *interfaceFluxes = uniformInterfacesFluxes.data() + N_FIELDS * i;

        // Sum owner fluxes
        std::size_t ownerRawId = solvedUniformInterfaceOwnerRawIds[i];
        double *ownerRHS = cellsRHS->rawData(ownerRawId);
        for (int k = 0; k < N_FIELDS; ++k) {
            ownerRHS[k] -= interfaceArea * interfaceFluxes[k];
        }

        // Sum neighbour fluxes
        std::size_t neighRawId = solvedUniformInterfaceNeighRawIds[i];
        double *neighRHS = cellsRHS->rawData(neighRawId);
        for (int k = 0; k < N_FIELDS; ++k) {
            neighRHS[k] += interfaceArea * interfaceFluxes[k];
        }
    }

    // Clean-up
    uniformInterfacesFluxes.cuda_free();

    uniformOwnerReconstructions.cuda_free();
    uniformNeighReconstructions.cuda_free();

    //
    // Process boundary interfaces
    //
    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceRawIds = computationInfo.getSolvedBoundaryInterfaceRawIds();
    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceSigns = computationInfo.getSolvedBoundaryInterfaceSigns();
    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceFluidRawIds = computationInfo.getSolvedBoundaryInterfaceFluidRawIds();
    const std::size_t nSolvedBoundaryInterfaces = solvedBoundaryInterfaceRawIds.size();

    // Evaluate interface reconstructions
    ScalarStorage<double> boundaryFluidReconstructions(N_FIELDS * nSolvedBoundaryInterfaces);
    ScalarStorage<double> boundaryVirtualReconstructions(N_FIELDS * nSolvedBoundaryInterfaces);
    boundaryFluidReconstructions.cuda_allocate();
    boundaryVirtualReconstructions.cuda_allocate();

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
        double *fluidReconstruction   = boundaryFluidReconstructions.data()   + N_FIELDS * i;
        double *virtualReconstruction = boundaryVirtualReconstructions.data() + N_FIELDS * i;

        reconstruction::eval(fluidRawId, computationInfo, order, interfaceCentroid, fluidMean, fluidReconstruction);
        evalInterfaceBCValues(problemType, interfaceBC, interfaceCentroid, interfaceNormal, fluidReconstruction, virtualReconstruction);
    }

    boundaryFluidReconstructions.cuda_updateDevice();
    boundaryVirtualReconstructions.cuda_updateDevice();

    // Evaluate fluxes
    ScalarStorage<double> boundaryInterfacesFluxes(N_FIELDS * nSolvedBoundaryInterfaces);
    boundaryInterfacesFluxes.cuda_allocate();

    double boundaryMaxEig;
    cuda_computeInterfaceFluxes(computationInfo, solvedBoundaryInterfaceRawIds, boundaryFluidReconstructions, boundaryVirtualReconstructions, &boundaryInterfacesFluxes, &boundaryMaxEig);

    // Update the residuals
    for (std::size_t i = 0; i < nSolvedBoundaryInterfaces; ++i) {
        // Info about the interface
        const std::size_t interfaceRawId = solvedBoundaryInterfaceRawIds[i];
        const double interfaceArea = computationInfo.rawGetInterfaceArea(interfaceRawId);
        const int interfaceSign = solvedBoundaryInterfaceSigns[i];
        const double *interfaceFluxes = boundaryInterfacesFluxes.data() + N_FIELDS * i;

        // Sum fluid fluxes
        std::size_t fluidRawId = solvedBoundaryInterfaceFluidRawIds[i];
        double *fluidRHS = cellsRHS->rawData(fluidRawId);
        for (int k = 0; k < N_FIELDS; ++k) {
            fluidRHS[k] -= interfaceSign * interfaceArea * interfaceFluxes[k];
        }
    }

    // Clean-up
    boundaryInterfacesFluxes.cuda_free();

    boundaryFluidReconstructions.cuda_free();
    boundaryVirtualReconstructions.cuda_free();

    // Evaluate maximum eigenvalue
    *maxEig = std::max(uniformMaxEig, boundaryMaxEig);
}

}
