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

#include "euler.hpp"
#include "reconstruction.hpp"
#include "utils.hpp"

using namespace bitpit;

namespace euler {

/*!
 * Computes approximate Riemann solver (Local Lax Friedrichs) for compressible perfect gas.
 *
 * \param conservativeL is the left conservative state
 * \param conservativeR is the right conservative state
 * \param n is the normal
 * \param[out] fluxes on output will contain the conservative fluxes
 * \param[out] lambda on output will contain the maximum eigenvalue
 */
void evalSplitting(const double *conservativeL, const double *conservativeR, const double *n, double *fluxes, double *lambda)
{
    // Primitive variables
    std::array<double, N_FIELDS> primitiveL;
    ::utils::conservative2primitive(conservativeL, primitiveL.data());

    std::array<double, N_FIELDS> primitiveR;
    ::utils::conservative2primitive(conservativeR, primitiveR.data());

    // Fluxes
    FluxData fL;
    evalFluxes(conservativeL, primitiveL.data(), n, fL.data());

    FluxData fR;
    evalFluxes(conservativeR, primitiveR.data(), n, fR.data());

    // Eigenvalues
    double unL     = ::utils::normalVelocity(primitiveL.data(), n);
    double aL      = std::sqrt(GAMMA * primitiveL[FID_T]);
    double lambdaL = std::abs(unL) + aL;

    double unR     = ::utils::normalVelocity(primitiveR.data(), n);
    double aR      = std::sqrt(GAMMA * primitiveR[FID_T]);
    double lambdaR = std::abs(unR) + aR;

    *lambda = std::max(lambdaR, lambdaL);

    // Splitting
    for (int k = 0; k < N_FIELDS; ++k) {
        fluxes[k] = 0.5 * ((fR[k] + fL[k]) - (*lambda) * (conservativeR[k] - conservativeL[k]));
    }
}

/*!
 * Calculates the conservative fluxes for a perfect gas.
 *
 * \param conservative is the conservative state
 * \param primitive is the primitive state
 * \param n is the normal direction
 * \param[out] fluxes on output will contain the conservative fluxes
 */
void evalFluxes(const double *conservative, const double *primitive, const double *n, double *fluxes)
{
    // Compute variables
    double u = primitive[FID_U];
    double v = primitive[FID_V];
    double w = primitive[FID_W];

    double vel2 = u * u + v * v + w * w;
    double un   = ::utils::normalVelocity(primitive, n);

    double p = primitive[FID_P];
    if (p < 0.) {
      log::cout() << "***** Negative pressure (" << p << ") in flux computation!\n";
    }

    double rho = conservative[FID_RHO];
    if (rho < 0.) {
        log::cout() << "***** Negative density in flux computation!\n";
    }

    double eto = p / (GAMMA - 1.) + 0.5 * rho * vel2;

    // Compute fluxes
    double massFlux = rho * un;

    fluxes[FID_EQ_C]   = massFlux;
    fluxes[FID_EQ_M_X] = massFlux * u + p * n[0];
    fluxes[FID_EQ_M_Y] = massFlux * v + p * n[1];
    fluxes[FID_EQ_M_Z] = massFlux * w + p * n[2];
    fluxes[FID_EQ_E]   = un * (eto + p);
}

/*!
 * Computes cell RHS.
 *
 * \param problemType is the problem type
 * \param computationInfo are the computation information
 * \param order is the order
 * \param solvedBoundaryInterfaceBCs is the storage for the interface boundary
 * conditions of the solved boundary cells
 * \param cellConservatives are the cell conservative values
 * \param[out] cellsRHS on output will containt the RHS
 * \param[out] maxEig on putput will containt the maximum eigenvalue
 */
void computeRHS(problem::ProblemType problemType, ComputationInfo &computationInfo,
                const int order, const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
                const ScalarPiercedStorageCollection<double> &cellConservatives, ScalarPiercedStorageCollection<double> *cellsRHS, double *maxEig)
{
    // Reset residuals
#if ENABLE_CUDA
    cuda_resetRHS(cellsRHS);
#else
    resetRHS(cellsRHS);
#endif

    // Update residuals
#if ENABLE_CUDA
    cuda_updateRHS(problemType, computationInfo, order, solvedBoundaryInterfaceBCs, cellConservatives,
                   cellsRHS, maxEig);
#else
    updateRHS(problemType, computationInfo, order, solvedBoundaryInterfaceBCs, cellConservatives,
              cellsRHS, maxEig);
#endif
}

/*!
 * Reset cell RHS.
 *
 * \param[in,out] rhs is the RHS that will be reset
 */
void resetRHS(ScalarPiercedStorageCollection<double> *cellsRHS)
{
    for (int k = 0; k < N_FIELDS; ++k) {
        (*cellsRHS)[k].fill(0.);
    }
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
void updateRHS(problem::ProblemType problemType, ComputationInfo &computationInfo,
               const int order, const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
               const ScalarPiercedStorageCollection<double> &cellConservatives, ScalarPiercedStorageCollection<double> *cellsRHS, double *maxEig)
{
    // Get mesh information
    const ScalarStorage<std::size_t> &solvedUniformInterfaceRawIds = computationInfo.getSolvedUniformInterfaceRawIds();
    const ScalarStorage<std::size_t> &solvedUniformInterfaceOwnerRawIds = computationInfo.getSolvedUniformInterfaceOwnerRawIds();
    const ScalarStorage<std::size_t> &solvedUniformInterfaceNeighRawIds = computationInfo.getSolvedUniformInterfaceNeighRawIds();
    const std::size_t nSolvedUniformInterfaces = solvedUniformInterfaceRawIds.size();

    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceRawIds = computationInfo.getSolvedBoundaryInterfaceRawIds();
    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceFluidRawIds = computationInfo.getSolvedBoundaryInterfaceFluidRawIds();
    const ScalarStorage<int> &solvedBoundaryInterfaceSigns = computationInfo.getSolvedBoundaryInterfaceSigns();
    const std::size_t nSolvedBoundaryInterfaces = solvedBoundaryInterfaceRawIds.size();

    // Update the residuals
    *maxEig = 0.0;

    // Process uniform interfaces
    for (std::size_t i = 0; i < nSolvedUniformInterfaces; ++i) {
        // Info about the interface
        const std::size_t interfaceRawId = solvedUniformInterfaceRawIds[i];
        const double interfaceArea = computationInfo.rawGetInterfaceArea(interfaceRawId);
        const std::array<double, 3> &interfaceNormal = computationInfo.rawGetInterfaceNormal(interfaceRawId);
        const std::array<double, 3> &interfaceCentroid = computationInfo.rawGetInterfaceCentroid(interfaceRawId);

        // Info about the interface owner
        std::size_t ownerRawId = solvedUniformInterfaceOwnerRawIds[i];

        std::array<double, N_FIELDS> ownerMean;
        for (int k = 0; k < N_FIELDS; ++k) {
            ownerMean[k] = cellConservatives[k].rawAt(ownerRawId);
        }

        // Info about the interface neighbour
        std::size_t neighRawId = solvedUniformInterfaceNeighRawIds[i];

        std::array<double, N_FIELDS> neighMean;
        for (int k = 0; k < N_FIELDS; ++k) {
            neighMean[k] = cellConservatives[k].rawAt(neighRawId);
        }

        // Evaluate interface reconstructions
        std::array<double, N_FIELDS> ownerReconstruction;
        std::array<double, N_FIELDS> neighReconstruction;

        reconstruction::eval(order, interfaceCentroid, ownerMean.data(), ownerReconstruction.data());
        reconstruction::eval(order, interfaceCentroid, neighMean.data(), neighReconstruction.data());

        // Evaluate the conservative fluxes
        FluxData fluxes;
        fluxes.fill(0.);

        double faceMaxEig;
        euler::evalSplitting(ownerReconstruction.data(), neighReconstruction.data(), interfaceNormal.data(), fluxes.data(), &faceMaxEig);

        *maxEig = std::max(faceMaxEig, *maxEig);

        // Sum the fluxes
        for (int k = 0; k < N_FIELDS; ++k) {
            (*cellsRHS)[k].rawAt(ownerRawId) -= interfaceArea * fluxes[k];
            (*cellsRHS)[k].rawAt(neighRawId) += interfaceArea * fluxes[k];
        }
    }

    // Process boundary interfaces
    for (std::size_t i = 0; i < nSolvedBoundaryInterfaces; ++i) {
        // Info about the interface
        const std::size_t interfaceRawId = solvedBoundaryInterfaceRawIds[i];
        const double interfaceArea = computationInfo.rawGetInterfaceArea(interfaceRawId);
        const int interfaceSign = solvedBoundaryInterfaceSigns[i];
        const std::array<double, 3> &interfaceNormal = computationInfo.rawGetInterfaceNormal(interfaceRawId);
        const std::array<double, 3> &interfaceCentroid = computationInfo.rawGetInterfaceCentroid(interfaceRawId);
        int interfaceBC = solvedBoundaryInterfaceBCs[i];

        // Info about the interface fluid cell
        std::size_t fluidRawId = solvedBoundaryInterfaceFluidRawIds[i];

        std::array<double, N_FIELDS> fluidMean;
        for (int k = 0; k < N_FIELDS; ++k) {
            fluidMean[k] = cellConservatives[k].rawAt(fluidRawId);
        }

        // Evaluate interface reconstructions
        std::array<double, N_FIELDS> fluidReconstruction;
        std::array<double, N_FIELDS> virtualReconstruction;

        reconstruction::eval(order, interfaceCentroid, fluidMean.data(), fluidReconstruction.data());
        evalInterfaceBCValues(interfaceCentroid, interfaceNormal, problemType, interfaceBC, fluidReconstruction.data(), virtualReconstruction.data());

        // Evaluate the conservative fluxes
        FluxData fluxes;
        fluxes.fill(0.);

        double faceMaxEig;
        euler::evalSplitting(fluidReconstruction.data(), virtualReconstruction.data(), interfaceNormal.data(), fluxes.data(), &faceMaxEig);

        *maxEig = std::max(faceMaxEig, *maxEig);

        // Sum the fluxes
        for (int k = 0; k < N_FIELDS; ++k) {
            (*cellsRHS)[k].rawAt(fluidRawId) -= interfaceSign * interfaceArea * fluxes[k];
        }
    }
}

/*!
 * Computes the boundary values for the specified interface.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param problemType is the type of problem being solved
 * \param BCType is the type of boundary condition to apply
 * \param innerValues are the inner values
 * \param[out] boundaryValues are the boundary values
 */
void evalInterfaceBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                           problem::ProblemType problemType, int BCType, const double *innerValues,
                           double *boundaryValues)
{
    std::array<double, BC_INFO_SIZE> info;
    problem::getBorderBCInfo(problemType, BCType, point, normal, info);

    switch (BCType)
    {
        case BC_FREE_FLOW:
            evalFreeFlowBCValues(point, normal, info, innerValues, boundaryValues);
            break;

        case BC_REFLECTING:
            evalReflectingBCValues(point, normal, info, innerValues, boundaryValues);
            break;

        case BC_WALL:
            evalWallBCValues(point, normal, info, innerValues, boundaryValues);
            break;

        case BC_DIRICHLET:
            evalDirichletBCValues(point, normal, info, innerValues, boundaryValues);
            break;

    }
}

/*!
 * Computes the boundary values for the free flow BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner values
 * \param[out] boundaryValues are the boundary values
 */
void evalFreeFlowBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                          const std::array<double, BC_INFO_SIZE> &info, const double *innerValues,
                          double *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);
    BITPIT_UNUSED(info);

    std::copy_n(innerValues, N_FIELDS, boundaryValues);
}

/*!
 * Computes the boundary values for the reflecting BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner values
 * \param[out] boundaryValues are the boundary values
 */
void evalReflectingBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                            const std::array<double, BC_INFO_SIZE> &info, const double *innerValues,
                            double *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(info);

    std::array<double, N_FIELDS> primitive;
    ::utils::conservative2primitive(innerValues, primitive.data());

    double u_n = ::utils::normalVelocity(primitive.data(), normal.data());

    primitive[FID_U] -= 2 * u_n * normal[0];
    primitive[FID_V] -= 2 * u_n * normal[1];
    primitive[FID_W] -= 2 * u_n * normal[2];

    ::utils::primitive2conservative(primitive.data(), boundaryValues);
}

/*!
 * Computes the boundary values for the wall BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner values
 * \param[out] boundaryValues are the boundary values
 */
void evalWallBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                      const std::array<double, BC_INFO_SIZE> &info, const double *innerValues,
                      double *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(info);

    evalReflectingBCValues(point, normal, info, innerValues, boundaryValues);
}

/*!
 * Computes the boundary values for the dirichlet BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param innerValues are the inner values
 * \param[out] boundaryValues are the boundary values
 */
void evalDirichletBCValues(const std::array<double, 3> &point, const std::array<double, 3> &normal,
                           const std::array<double, BC_INFO_SIZE> &info, const double *innerValues,
                           double *boundaryValues)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);
    BITPIT_UNUSED(innerValues);

    ::utils::primitive2conservative(info.data(), boundaryValues);
}

}
