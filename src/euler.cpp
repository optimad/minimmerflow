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
void evalSplitting(const double *conservativeL, const double *conservativeR, const std::array<double, 3> &n, FluxData *fluxes, double *lambda)
{
    // Primitive variables
    std::array<double, N_FIELDS> primitiveL;
    ::utils::conservative2primitive(conservativeL, primitiveL.data());

    std::array<double, N_FIELDS> primitiveR;
    ::utils::conservative2primitive(conservativeR, primitiveR.data());

    // Fluxes
    FluxData fL;
    evalFluxes(conservativeL, primitiveL.data(), n, &fL);

    FluxData fR;
    evalFluxes(conservativeR, primitiveR.data(), n, &fR);

    // Eigenvalues
    double unL     = ::utils::normalVelocity(primitiveL.data(), n.data());
    double aL      = std::sqrt(GAMMA * primitiveL[FID_T]);
    double lambdaL = std::abs(unL) + aL;

    double unR     = ::utils::normalVelocity(primitiveR.data(), n.data());
    double aR      = std::sqrt(GAMMA * primitiveR[FID_T]);
    double lambdaR = std::abs(unR) + aR;

    *lambda = std::max(lambdaR, lambdaL);

    // Splitting
    for (int k = 0; k < N_FIELDS; ++k) {
        (*fluxes)[k] = 0.5 * ((fR[k] + fL[k]) - (*lambda) * (conservativeR[k] - conservativeL[k]));
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
void evalFluxes(const double *conservative, const double *primitive, const std::array<double, 3> &n, FluxData *fluxes)
{
    // Compute variables
    double u = primitive[FID_U];
    double v = primitive[FID_V];
    double w = primitive[FID_W];

    double vel2 = u * u + v * v + w * w;
    double un   = ::utils::normalVelocity(primitive, n.data());

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

    (*fluxes)[FID_EQ_C]   = massFlux;
    (*fluxes)[FID_EQ_M_X] = massFlux * u + p * n[0];
    (*fluxes)[FID_EQ_M_Y] = massFlux * v + p * n[1];
    (*fluxes)[FID_EQ_M_Z] = massFlux * w + p * n[2];
    (*fluxes)[FID_EQ_E]   = un * (eto + p);
}

/*!
 * Computes flux differences for every cell.
 *
 * \param problemType is the problem type
 * \param meshInfo are the geometrical information
 * \param cellSolvedFlag is the storage for the cell solved flag
 * \param order is the order
 * \param cellConservatives are the cell conservative values
 * \param interfaceBCs is the boundary conditions storage
 * \param[out] cellsRHS on output will containt the RHS
 * \param[out] maxEig on putput will containt the maximum eigenvalue
 */
void computeRHS(problem::ProblemType problemType, const MeshGeometricalInfo &meshInfo,
                const ScalarPiercedStorage<bool> &cellSolvedFlag, const int order,
                const ScalarPiercedStorage<double> &cellConservatives, const ScalarPiercedStorage<int> &interfaceBCs,
                ScalarPiercedStorage<double> *cellsRHS, double *maxEig)
{
    // Get mesh information
    const VolumeKernel &mesh = meshInfo.getPatch();

    const ScalarStorage<std::size_t> &internalCellRawIds = meshInfo.getCellRawIds();
    const std::size_t nInternalCells = internalCellRawIds.size();

    const ScalarStorage<std::size_t> &interfaceRawIds = meshInfo.getInterfaceRawIds();
    const std::size_t nInterfaces = interfaceRawIds.size();

    // Reset the residuals
    for (std::size_t i = 0; i < nInternalCells; ++i) {
        const std::size_t cellRawId = internalCellRawIds[i];
        double *cellRHS = cellsRHS->rawData(cellRawId);
        for (int k = 0; k < N_FIELDS; ++k) {
            cellRHS[k] = 0.;
        }
    }

    // Update the residuals
    *maxEig = 0.0;

    for (std::size_t i = 0; i < nInterfaces; ++i) {
        const std::size_t interfaceRawId = interfaceRawIds[i];
        const Interface &interface = mesh.getInterfaces().rawAt(interfaceRawId);

        // Info about the interface owner
        long ownerId = interface.getOwner();
        VolumeKernel::CellConstIterator ownerItr = mesh.getCellConstIterator(ownerId);
        std::size_t ownerRawId = ownerItr.getRawIndex();
        const double *ownerMean = cellConservatives.rawData(ownerRawId);
        double *ownerRHS = cellsRHS->rawData(ownerRawId);
        bool ownerSolved = cellSolvedFlag.rawAt(ownerRawId);

        // Info about the interface neighbour
        long neighId = interface.getNeigh();
        std::size_t neighRawId = std::numeric_limits<std::size_t>::max();
        const double *neighMean = nullptr;
        double *neighRHS = nullptr;
        bool neighSolved = false;
        if (neighId >= 0) {
            VolumeKernel::CellConstIterator neighItr = mesh.getCellConstIterator(neighId);

            neighRawId  = neighItr.getRawIndex();
            neighMean   = cellConservatives.rawData(neighRawId);
            neighRHS    = cellsRHS->rawData(neighRawId);
            neighSolved = cellSolvedFlag.rawAt(neighRawId);
        }

        // Check if the interface needs to be solved
        if (!ownerSolved && !neighSolved) {
            continue;
        }

        // Info about the interface
        int interfaceBCType = interfaceBCs.rawAt(interfaceRawId);
        const double interfaceArea = meshInfo.rawGetInterfaceArea(interfaceRawId);
        const std::array<double, 3> &interfaceNormal = meshInfo.rawGetInterfaceNormal(interfaceRawId);
        const std::array<double, 3> &interfaceCentroid = meshInfo.rawGetInterfaceCentroid(interfaceRawId);

        // Evaluate the conservative fluxes
        std::array<double, N_FIELDS> ownerReconstruction;
        std::array<double, N_FIELDS> neighReconstruction;
        if (interfaceBCType == BC_NONE)  {
            reconstruction::eval(ownerRawId, meshInfo, order, interfaceCentroid, ownerMean, ownerReconstruction.data());
            reconstruction::eval(neighRawId, meshInfo, order, interfaceCentroid, neighMean, neighReconstruction.data());
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
                fluidReconstruction   = ownerReconstruction.data();
                virtualReconstruction = neighReconstruction.data();
            } else {
                flipNormal            = true;
                fluidRawId            = neighRawId;
                fluidMean             = neighMean;
                fluidReconstruction   = neighReconstruction.data();
                virtualReconstruction = ownerReconstruction.data();
            }

            int interfaceBC = interfaceBCs.rawAt(interfaceRawId);

            std::array<double, 3> interfaceNormal = meshInfo.rawGetInterfaceNormal(interfaceRawId);
            if (flipNormal) {
                interfaceNormal = -1. * interfaceNormal;
            }

            reconstruction::eval(fluidRawId, meshInfo, order, interfaceCentroid, fluidMean, fluidReconstruction);
            evalInterfaceBCValues(problemType, interfaceBC, interfaceCentroid, interfaceNormal, fluidReconstruction, virtualReconstruction);
        }

        FluxData fluxes;
        fluxes.fill(0.);

        double faceMaxEig;
        euler::evalSplitting(ownerReconstruction.data(), neighReconstruction.data(), interfaceNormal, &fluxes, &faceMaxEig);

        *maxEig = std::max(faceMaxEig, *maxEig);

        // Sum the fluxes
        if (ownerSolved)  {
            for (int k = 0; k < N_FIELDS; ++k) {
                ownerRHS[k] -= interfaceArea * fluxes[k];
            }
        }

        if (neighSolved)  {
            for (int k = 0; k < N_FIELDS; ++k) {
                neighRHS[k] += interfaceArea * fluxes[k];
            }
        }
    }
}

/*!
 * Computes the boundary values for the specified interface.
 *
 * \param problemType is the type of problem being solved
 * \param BCType is the type of boundary condition to apply
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param conservative are the inner conservative values
 * \param[out] conservative_BC are the outer conservative values
 */
void evalInterfaceBCValues(problem::ProblemType problemType, int BCType,
                           const std::array<double, 3> &point,
                           const std::array<double, 3> &normal,
                           const double *conservative, double *conservative_BC)
{
    std::array<double, BC_INFO_SIZE> info;
    problem::getBorderBCInfo(problemType, BCType, point, normal, info);

    switch (BCType)
    {
        case BC_FREE_FLOW:
            evalFreeFlowBCValues(point, normal, info, conservative, conservative_BC);
            break;

        case BC_REFLECTING:
            evalReflectingBCValues(point, normal, info, conservative, conservative_BC);
            break;

        case BC_WALL:
            evalWallBCValues(point, normal, info, conservative, conservative_BC);
            break;

        case BC_DIRICHLET:
            evalDirichletBCValues(point, normal, info, conservative, conservative_BC);
            break;

    }
}

/*!
 * Computes the boundary values for the free flow BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param[out] conservative_BC are the outer conservative values
 */
void evalFreeFlowBCValues(const std::array<double, 3> &point,
                          const std::array<double, 3> &normal,
                          const std::array<double, BC_INFO_SIZE> &info,
                          const double *conservative, double *conservative_BC)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);
    BITPIT_UNUSED(info);

    std::copy_n(conservative, N_FIELDS, conservative_BC);
}

/*!
 * Computes the boundary values for the reflecting BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param[out] conservative_BC are the outer conservative values
 */
void evalReflectingBCValues(const std::array<double, 3> &point,
                            const std::array<double, 3> &normal,
                            const std::array<double, BC_INFO_SIZE> &info,
                            const double *conservative, double *conservative_BC)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(info);

    std::array<double, N_FIELDS> primitive;
    ::utils::conservative2primitive(conservative, primitive.data());

    std::array<double, 3> u    = {{primitive[FID_U], primitive[FID_V], primitive[FID_W]}};
    std::array<double, 3> u_n  = ::utils::normalVelocity(primitive.data(), normal.data()) * normal;

    primitive[FID_U] = u[0] - 2 * u_n[0];
    primitive[FID_V] = u[1] - 2 * u_n[1];
    primitive[FID_W] = u[2] - 2 * u_n[2];

    ::utils::primitive2conservative(primitive.data(), conservative_BC);
}

/*!
 * Computes the boundary values for the wall BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param[out] conservative_BC are the outer conservative values
 */
void evalWallBCValues(const std::array<double, 3> &point,
                      const std::array<double, 3> &normal,
                      const std::array<double, BC_INFO_SIZE> &info,
                      const double *conservative, double *conservative_BC)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(info);

    evalReflectingBCValues(point, normal, info, conservative, conservative_BC);
}

/*!
 * Computes the boundary values for the dirichlet BC.
 *
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param info are the info needed for evaluating the boundary condition
 * \param[out] conservative_BC are the outer conservative values
 */
void evalDirichletBCValues(const std::array<double, 3> &point,
                           const std::array<double, 3> &normal,
                           const std::array<double, BC_INFO_SIZE> &info,
                           const double *conservative, double *conservative_BC)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);
    BITPIT_UNUSED(conservative);

    ::utils::primitive2conservative(info.data(), conservative_BC);
}

}
