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

namespace euler {

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
 * \param nInterfaces is the number of interfaces
 * \param interfaceRawIds are the raw ids of the interfaces
 * \param interfaceSolvedFlag are the solved flags of the interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param ownerReconstructions are the owner reconstructions
 * \param neighReconstructions are the neighbour reconstructions
 * \param[out] interfacesFluxes on output will containt the interface fluxes
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
__global__ void dev_computeInterfaceFluxes(std::size_t nInterfaces, const std::size_t *interfaceRawIds, const int *interfaceSolvedFlag, const double *interfaceNormals,
                                           const double *ownerReconstructions, const double *neighReconstructions,
                                           double *interfacesFluxes, double *interfacesMaxEig)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];

    // Check if the interface needs to be solved
    if (!interfaceSolvedFlag[interfaceRawId]) {
        return;
    }

    // Info about the interface
    const double *interfaceNormal = interfaceNormals + 3 * interfaceRawId;

    // Evaluate the conservative fluxes
    const double *ownerReconstruction = ownerReconstructions + N_FIELDS * interfaceRawId;
    const double *neighReconstruction = neighReconstructions + N_FIELDS * interfaceRawId;

    double *interfaceFluxes = interfacesFluxes + N_FIELDS * interfaceRawId;
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double *interfaceMaxEig = interfacesMaxEig + interfaceRawId;
    *interfaceMaxEig = 0.;

    dev_evalSplitting(ownerReconstruction, neighReconstruction, interfaceNormal, interfaceFluxes, interfaceMaxEig);
}

/*!
 * Compute interface fluxes.
 *
 * \param meshInfo are the geometrical information
 * \param interfaceSolvedFlag is the storage for the interface solved flag
 * \param ownerReconstructions is the storage fot the owner reconstructions
 * \param neighReconstructions is the storage fot the neighbour reconstructions
 * \param interfaceSolvedFlag is the storage for the interface solved flag
 * \param[out] interfacesFluxes on output will containt the interface fluxes
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
void cuda_computeInterfaceFluxes(const MeshGeometricalInfo &meshInfo, const ScalarPiercedStorage<int> &interfaceSolvedFlag,
                                 const ScalarPiercedStorage<double> &ownerReconstructions, const ScalarPiercedStorage<double> &neighReconstructions,
                                 ScalarPiercedStorage<double> *interfacesFluxes, double *maxEig)
{
    // Get mesh information
    const bitpit::VolumeKernel &mesh = meshInfo.getPatch();

    const ScalarStorage<std::size_t> &interfaceRawIds = meshInfo.getInterfaceRawIds();
    const std::size_t nInterfaces = interfaceRawIds.size();

    // Initialize device data
    ScalarPiercedStorage<double> interfacesMaxEig(1, &mesh.getInterfaces());
    interfacesMaxEig.cuda_allocate();

    // Evaluate fluxes
    const int *devInterfaceSolvedFlag = interfaceSolvedFlag.cuda_devData();
    const std::size_t *devInterfaceRawIds = meshInfo.cuda_getInterfaceRawIdDevData();
    const double *devInterfaceNormals = meshInfo.cuda_getInterfaceNormalDevData();

    const double *devOwnerReconstructions = ownerReconstructions.cuda_devData();
    const double *devNeighReconstructions = neighReconstructions.cuda_devData();

    double *devInterfacesFluxes = interfacesFluxes->cuda_devData();
    double *devInterfacesMaxEig = interfacesMaxEig.cuda_devData();

    int blockSize = 256;
    int numBlocks = (nInterfaces + blockSize - 1) / blockSize;
    dev_computeInterfaceFluxes<<<numBlocks, blockSize>>>(nInterfaces, devInterfaceRawIds, devInterfaceSolvedFlag, devInterfaceNormals,
                                                         devOwnerReconstructions, devNeighReconstructions,
                                                         devInterfacesFluxes, devInterfacesMaxEig);

    // Update host data
    interfacesFluxes->cuda_updateHost();
    interfacesMaxEig.cuda_updateHost();

    // Compute maximum eigenvlaue
    *maxEig = 0.;
    for (std::size_t i = 0; i < nInterfaces; ++i) {
        const std::size_t interfaceRawId = interfaceRawIds[i];
        if (!interfaceSolvedFlag.rawAt(interfaceRawId)) {
            continue;
        }

        const double interfaceMaxEig = interfacesMaxEig.rawAt(interfaceRawId);
        *maxEig = std::max(interfaceMaxEig, *maxEig);
    }

    // Finalize data
    interfacesMaxEig.cuda_free();
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
 * Computes flux differences for every cell.
 *
 * \param problemType is the problem type
 * \param meshInfo are the geometrical information
 * \param cellSolvedFlag is the storage for the cell solved flag
 * \param interfaceSolvedFlag is the storage for the interface solved flag
 * \param order is the order
 * \param cellConservatives are the cell conservative values
 * \param interfaceBCs is the boundary conditions storage
 * \param[out] cellsRHS on output will containt the RHS
 * \param[out] maxEig on putput will containt the maximum eigenvalue
 */
void cuda_updateRHS(problem::ProblemType problemType, const MeshGeometricalInfo &meshInfo,
                    const ScalarPiercedStorage<int> &cellSolvedFlag, const ScalarPiercedStorage<int> &interfaceSolvedFlag,
                    const int order, const ScalarPiercedStorage<double> &cellConservatives,
                    const ScalarPiercedStorage<int> &interfaceBCs, ScalarPiercedStorage<double> *cellsRHS, double *maxEig)
{
    // Get mesh information
    const bitpit::VolumeKernel &mesh = meshInfo.getPatch();

    const ScalarStorage<std::size_t> &interfaceRawIds = meshInfo.getInterfaceRawIds();
    const std::size_t nInterfaces = interfaceRawIds.size();

    // Evaluate interface reconstructions
    ScalarPiercedStorage<double> ownerReconstructions(N_FIELDS, &mesh.getInterfaces());
    ScalarPiercedStorage<double> neighReconstructions(N_FIELDS, &mesh.getInterfaces());
    ownerReconstructions.cuda_allocate();
    neighReconstructions.cuda_allocate();

    for (std::size_t i = 0; i < nInterfaces; ++i) {
        const std::size_t interfaceRawId = interfaceRawIds[i];
        if (!interfaceSolvedFlag.rawAt(interfaceRawId)) {
            continue;
        }

        // Info about the interface
        const bitpit::Interface &interface = mesh.getInterfaces().rawAt(interfaceRawId);
        int interfaceBCType = interfaceBCs.rawAt(interfaceRawId);
        const std::array<double, 3> &interfaceCentroid = meshInfo.rawGetInterfaceCentroid(interfaceRawId);

        // Info about the interface owner
        long ownerId = interface.getOwner();
        bitpit::VolumeKernel::CellConstIterator ownerItr = mesh.getCellConstIterator(ownerId);
        std::size_t ownerRawId = ownerItr.getRawIndex();
        const double *ownerMean = cellConservatives.rawData(ownerRawId);
        bool ownerSolved = cellSolvedFlag.rawAt(ownerRawId);

        // Info about the interface neighbour
        long neighId = interface.getNeigh();
        std::size_t neighRawId = std::numeric_limits<std::size_t>::max();
        const double *neighMean = nullptr;
        if (neighId >= 0) {
            bitpit::VolumeKernel::CellConstIterator neighItr = mesh.getCellConstIterator(neighId);

            neighRawId  = neighItr.getRawIndex();
            neighMean   = cellConservatives.rawData(neighRawId);
        }

        // Evaluate reconstructions
        double *ownerReconstruction = ownerReconstructions.rawData(interfaceRawId);
        double *neighReconstruction = neighReconstructions.rawData(interfaceRawId);

        if (interfaceBCType == BC_NONE)  {
            reconstruction::eval(ownerRawId, meshInfo, order, interfaceCentroid, ownerMean, ownerReconstruction);
            reconstruction::eval(neighRawId, meshInfo, order, interfaceCentroid, neighMean, neighReconstruction);
        } else {
            long fluidRawId;
            bool flipNormal;
            const double *fluidMean;
            double *fluidReconstruction;
            double *virtualReconstruction;
            if (ownerSolved) {
                flipNormal            = false;
                fluidRawId            = ownerRawId;
                fluidMean             = ownerMean;
                fluidReconstruction   = ownerReconstruction;
                virtualReconstruction = neighReconstruction;
            } else {
                flipNormal            = true;
                fluidRawId            = neighRawId;
                fluidMean             = neighMean;
                fluidReconstruction   = neighReconstruction;
                virtualReconstruction = ownerReconstruction;
            }

            int interfaceBC = interfaceBCs.rawAt(interfaceRawId);

            std::array<double, 3> boundaryNormal = meshInfo.rawGetInterfaceNormal(interfaceRawId);
            if (flipNormal) {
                boundaryNormal = -1. * boundaryNormal;
            }

            reconstruction::eval(fluidRawId, meshInfo, order, interfaceCentroid, fluidMean, fluidReconstruction);
            evalInterfaceBCValues(problemType, interfaceBC, interfaceCentroid, boundaryNormal, fluidReconstruction, virtualReconstruction);
        }
    }

    ownerReconstructions.cuda_updateDevice();
    neighReconstructions.cuda_updateDevice();

    // Evaluate fluxes
    ScalarPiercedStorage<double> interfacesFluxes(N_FIELDS, &mesh.getInterfaces());
    interfacesFluxes.cuda_allocate();

    cuda_computeInterfaceFluxes(meshInfo, interfaceSolvedFlag, ownerReconstructions, neighReconstructions, &interfacesFluxes, maxEig);

    // Reset the residuals
    cuda_resetRHS(cellsRHS);

    // Update the residuals
    for (std::size_t i = 0; i < nInterfaces; ++i) {
        const std::size_t interfaceRawId = interfaceRawIds[i];
        if (!interfaceSolvedFlag.rawAt(interfaceRawId)) {
            continue;
        }

        // Info about the interface
        const bitpit::Interface &interface = mesh.getInterfaces().rawAt(interfaceRawId);
        const double interfaceArea = meshInfo.rawGetInterfaceArea(interfaceRawId);
        const double *interfaceFluxes = interfacesFluxes.rawData(interfaceRawId);

        // Sum owner fluxes
        long ownerId = interface.getOwner();
        bitpit::VolumeKernel::CellConstIterator ownerItr = mesh.getCellConstIterator(ownerId);
        std::size_t ownerRawId = ownerItr.getRawIndex();
        double *ownerRHS = cellsRHS->rawData(ownerRawId);
        bool ownerSolved = cellSolvedFlag.rawAt(ownerRawId);
        if (ownerSolved)  {
            for (int k = 0; k < N_FIELDS; ++k) {
                ownerRHS[k] -= interfaceArea * interfaceFluxes[k];
            }
        }

        // Sum neighbour fluxes
        long neighId = interface.getNeigh();
        if (neighId >= 0) {
            bitpit::VolumeKernel::CellConstIterator neighItr = mesh.getCellConstIterator(neighId);
            std::size_t neighRawId = neighItr.getRawIndex();
            bool neighSolved = cellSolvedFlag.rawAt(neighRawId);
            if (neighSolved)  {
                double *neighRHS = cellsRHS->rawData(neighRawId);
                for (int k = 0; k < N_FIELDS; ++k) {
                    neighRHS[k] += interfaceArea * interfaceFluxes[k];
                }
            }
        }
    }

    // Clean-up
    interfacesFluxes.cuda_free();

    ownerReconstructions.cuda_free();
    neighReconstructions.cuda_free();
}

}
