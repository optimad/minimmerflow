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

#include "problem.hpp"
#include "utils.hpp"

using namespace bitpit;

namespace problem {

/*!
 * Returns the problem type.
 *
 * \result The problem type.
 */
ProblemType getProblemType()
{
    std::string problemKey = (std::string) config::root["problem"].get<std::string>("type");
    if (problemKey == "vortex_xy") {
        return PROBLEM_VORTEX_XY;
    } else if (problemKey == "vortex_zx") {
        return PROBLEM_VORTEX_ZX;
    } else if (problemKey == "vortex_yz") {
        return PROBLEM_VORTEX_YZ;
    } else if (problemKey == "radsod") {
        return PROBLEM_RADSOD;
    } else if (problemKey == "sod3d_x") {
        return PROBLEM_SOD_X;
    } else if (problemKey == "sod3d_y") {
        return PROBLEM_SOD_Y;
    } else if (problemKey == "sod3d_z") {
        return PROBLEM_SOD_Z;
    } else if (problemKey == "ffstep") {
        return PROBLEM_FFSTEP;
    } else {
        throw std::runtime_error("Problem " + problemKey + " is not supported.");
    }
}

/*!
 * Returns domain data for a problem.
 *
 * \param problemType is the type of problem to solve
 * \param[out] dimensions is the number of geometriacal dimensions of the
 * problem
 * \param[out] origin on output will contain the origin of the domain
 * \param[out] length on output will contain length of the domain
 */
void getDomainData(ProblemType problemType, int &dimensions, std::array<double, 3> *origin, double *length)
{
    // Dimensions
    switch (problemType) {

    case PROBLEM_VORTEX_ZX:
    case PROBLEM_VORTEX_YZ:
        dimensions = 3;
        break;

    default:
        dimensions = config::root["problem"].get<int>("dimensions", 2.);
        break;

    }

    // Origin
    bool customOrigin = false;
    if (config::root["problem"].hasSection("domain")) {
        customOrigin = config::root["problem"]["domain"].hasSection("origin");
    }

    if (customOrigin) {
        (*origin)[0] = config::root["problem"]["domain"]["origin"].get<double>("x", 0.);
        (*origin)[1] = config::root["problem"]["domain"]["origin"].get<double>("y", 0.);
        (*origin)[2] = config::root["problem"]["domain"]["origin"].get<double>("z", 0.);
    } else {
        switch (problemType) {

        case PROBLEM_VORTEX_XY:
        case PROBLEM_VORTEX_ZX:
        case PROBLEM_VORTEX_YZ:
            (*origin) = {{-5, -5, -5.}};
            break;

        case PROBLEM_SOD_X:
        case PROBLEM_SOD_Y:
        case PROBLEM_SOD_Z:
            (*origin) = {{-1, -1, -1.}};
            break;

        default:
            (*origin) = {{0, 0, 0}};
            break;
        }
    }

    // Length
    bool customLength = false;
    if (config::root["problem"].hasSection("domain")) {
        customLength = config::root["problem"]["domain"].hasOption("length");
    }

    if (customLength) {
        (*length) = config::root["problem"]["domain"].get<double>("length");
    } else {
        switch (problemType) {

        case PROBLEM_VORTEX_XY:
        case PROBLEM_VORTEX_ZX:
        case PROBLEM_VORTEX_YZ:
            (*length) = 10.;
            break;

        case PROBLEM_SOD_X:
        case PROBLEM_SOD_Y:
        case PROBLEM_SOD_Z:
            (*length) = 2.;
            break;

        case PROBLEM_RADSOD:
            (*length) = 8.;
            break;

        default:
            throw std::runtime_error("The lenght of the domain is manditory.");

        }
    }
}

/*!
 * Returns the start time of the simulation.
 *
 * \param problemType is the type of problem to solve
 * \param[out] dimensions is the number of geometriacal dimensions of the
 * problem
 * \result The start time of the simulation.
 */
double getStartTime(ProblemType problemType, int dimensions)
{
    BITPIT_UNUSED(problemType);
    BITPIT_UNUSED(dimensions);

    return config::root["problem"].get<int>("start", 0.);
}

/*!
 * Returns the end time of the simulation.
 *
 * \param problemType is the type of problem to solve
 * \param[out] dimensions is the number of geometriacal dimensions of the
 * problem
 * \result The end time of the simulation.
 */
double getEndTime(ProblemType problemType, int dimensions)
{
    bool customTime = false;
    if (config::root["problem"].hasSection("time")) {
        customTime = config::root["problem"]["time"].hasOption("end");
    }

    if (customTime) {
        return config::root["problem"]["time"].get<double>("end");
    } else {
        switch (problemType) {

        case PROBLEM_VORTEX_XY:
        case PROBLEM_VORTEX_ZX:
        case PROBLEM_VORTEX_YZ:
            if (dimensions == 2) {
                return 2.;
            } else {
                return 1.;
            }
            break;

        case PROBLEM_SOD_X:
        case PROBLEM_SOD_Y:
        case PROBLEM_SOD_Z:
            return 0.3;

        case PROBLEM_RADSOD:
            return 4.;

        case PROBLEM_FFSTEP:
            return 4.;

        }
    }

    BITPIT_UNREACHABLE("Problem type not supported");
}

/*!
 * Computes initial values for the specified cell.
 *
 * \param problemType is the type of problem to solve
 * \param cell is the cell
 * \param[out] conservatives are the initial values for the specified cell
 */
void evalCellInitalConservatives(ProblemType problemType, const bitpit::Cell &cell, const MeshGeometricalInfo &meshInfo, double *conservatives)
{
    evalCellExactConservatives(problemType, cell, meshInfo, 0.0, conservatives);
}

/*!
 * Computes the exact solution for the specified cell at the specified time.
 *
 * \param problemType is the type of problem to solve
 * \param cell is the cell
 * \param t is the time at which the solution is requested
 * \param[out] conservatives are the exact solution values for the specified
 * cell
 */
void evalCellExactConservatives(ProblemType problemType, const bitpit::Cell &cell, const MeshGeometricalInfo &meshInfo, double t, double *conservatives)
{
    problem::evalExactConservatives(problemType, meshInfo.getDimension(), meshInfo.getCellCentroid(cell.getId()), t, conservatives);
}

/*!
 * Computes the exact solution for the specified cell at the specified time.
 *
 * \param problemType is the type of problem to solve
 * \param point is the point where the exact solution will be evaluated
 * \param t is the time at which the solution is requested
 * \param[out] conservatives are the exact solution values for the specified cell
 */
void evalExactConservatives(ProblemType problemType, int dimensions, std::array<double, 3> point, double t, double *conservatives)
{
    std::array<double, N_FIELDS> primitives;
    switch (problemType){

        case PROBLEM_VORTEX_XY:
        case PROBLEM_VORTEX_ZX:
        case PROBLEM_VORTEX_YZ:
        {
            // Vortex data
            const double VORTEX_BETA  = 5.0;
            const double VORTEX_P_INF = 1.0;
            const double VORTEX_T_INF = 1.0;
            double VORTEX_U_INF = 1.0;
            double VORTEX_V_INF = 1.0;
            double VORTEX_W_INF = 1.0;

            if (problemType == PROBLEM_VORTEX_XY) {
                VORTEX_W_INF = 0.;
            } else if (problemType == PROBLEM_VORTEX_ZX) {
                VORTEX_V_INF = 0.;
            } else if (problemType == PROBLEM_VORTEX_YZ) {
                VORTEX_U_INF = 0.;
            }

            // Position of point at time t
            point[0] -= t * VORTEX_U_INF;
            point[1] -= t * VORTEX_V_INF;
            point[2] -= t * VORTEX_W_INF;

            // Coordinates used to evaluate vortex data
            double csi = 0.;
            double eta = 0.;
            if (problemType == PROBLEM_VORTEX_XY) {
                csi = point[0];
                eta = point[1];
            } else if (problemType == PROBLEM_VORTEX_ZX) {
                csi = point[2];
                eta = point[0];
            } else if (problemType == PROBLEM_VORTEX_YZ) {
                csi = point[1];
                eta = point[2];
            }

            // Vortex data
            double r     = std::sqrt(csi * csi + eta * eta);
            double shape = VORTEX_BETA / (2 * M_PI) * std::exp(0.5 * (1 - r * r));

            double du = 0.;
            double dv = 0.;
            double dw = 0.;
            if (problemType == PROBLEM_VORTEX_XY) {
                du = - eta * shape;
                dv =   csi * shape;
                dw = 0.;
            } else if (problemType == PROBLEM_VORTEX_ZX) {
                dw = - eta * shape;
                du =   csi * shape;
                dv = 0.;
            } else if (problemType == PROBLEM_VORTEX_YZ) {
                dv = - eta * shape;
                dw =   csi * shape;
                du = 0.;
            }

            const double dT = - (GAMMA - 1.0) / (2 * GAMMA) * shape * shape;

            const double dp = std::pow(VORTEX_T_INF + dT, GAMMA / (GAMMA - 1.0)) - 1.0;

            primitives[FID_P] = VORTEX_P_INF + dp;
            primitives[FID_U] = VORTEX_U_INF + du;
            primitives[FID_V] = VORTEX_V_INF + dv;
            if (dimensions == 3) {
                primitives[FID_W] = VORTEX_W_INF + dw;
            } else {
                primitives[FID_W] = 0;
            }
            primitives[FID_T] = VORTEX_T_INF + dT;

            break;
        }

        case PROBLEM_RADSOD:
        {
            double r = sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);

            primitives[FID_U] = 0.0;
            primitives[FID_V] = 0.0;
            primitives[FID_W] = 0.0;
            if (r < 1.) {
                primitives[FID_P] = 1.0;
                primitives[FID_T] = 1.0 / 1.0;
            } else {
                primitives[FID_P] = 0.1;
                primitives[FID_T] = 0.1 / 0.125;
            }

            break;
        }

        case PROBLEM_SOD_X:
        case PROBLEM_SOD_Y:
        case PROBLEM_SOD_Z:
        {
            primitives[FID_U] = 0.0;
            primitives[FID_V] = 0.0;
            primitives[FID_W] = 0.0;

            double r;
            switch (problemType){

                case PROBLEM_SOD_X:
                    r = point[0];
                    break;

                case PROBLEM_SOD_Y:
                    r = point[1];
                    break;

                default:
                    r = point[2];

            }

            if (r < 0.) {
                primitives[FID_P] = 1.0;
                primitives[FID_T] = 1.0 / 1.0;
            } else {
                primitives[FID_P] = 0.1;
                primitives[FID_T] = 0.1 / 0.125;
            }

            break;
        }
        case PROBLEM_FFSTEP:
        {
            primitives[FID_U] = 3.0;
            primitives[FID_V] = 0.0;
            primitives[FID_W] = 0.0;
            primitives[FID_P] = 1.0;
            primitives[FID_T] = 1.0 / 1.4;
        }
    }

    ::utils::primitive2conservative(primitives.data(), conservatives);
}

/*!
 * Gets the type of boundary conditions associated to the specified border
 * interface.
 *
 * \param problemType is the type of problem to solve
 * \param id is the id of the border interface
 * \param meshInfo are the information about the mesh
 * \result The the type of boundary conditions associated to the specified
 * border interface.
 */
int getBorderBCType(ProblemType problemType, long id, const MeshGeometricalInfo &meshInfo)
{
    BITPIT_UNUSED(id);
    BITPIT_UNUSED(meshInfo);

    switch (problemType) {

    case (PROBLEM_VORTEX_XY):
    case (PROBLEM_VORTEX_ZX):
    case (PROBLEM_VORTEX_YZ):
    case (PROBLEM_SOD_X):
    case (PROBLEM_SOD_Y):
    case (PROBLEM_SOD_Z):
        return BC_FREE_FLOW;

    case (PROBLEM_RADSOD):
        return BC_REFLECTING;

    case (PROBLEM_FFSTEP):
    {
        std::array<double, 3> faceCentroid = meshInfo.getInterfaceCentroid(id);
        if (faceCentroid[0] < 1e-10) {
            return BC_DIRICHLET;
        } else if (faceCentroid[0] > 3.2 - 1e-10) {
            return BC_FREE_FLOW;
        } else {
            return BC_REFLECTING;
        }
    }

    default:
        BITPIT_UNREACHABLE("Problem type not supported");

    }
}

/*!
 * Gets the information needed for evaluating the boundary condition associated
 * to the specified border interface.
 *
 * \param problemType is the type of problem being solved
 * \param BCType is the type of boundary condition to apply
 * \param point is the point where the boundary condition should be applied
 * \param normal is the normal needed for evaluating the boundary condition
 * \param[out] info on output will contain the needed information
 */
void getBorderBCInfo(ProblemType problemType, int BCType, const std::array<double, 3> &point,
                     const std::array<double, 3> &normal, std::array<double, BC_INFO_SIZE> &info)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);

    switch (problemType) {

    case (PROBLEM_FFSTEP):
    {
        switch (BCType) {

        case (BC_DIRICHLET):
            info[FID_U] = 3.0;
            info[FID_V] = 0.;
            info[FID_W] = 0.;
            info[FID_P] = 1.;
            info[FID_T] = 1./1.4;
            return;

        }
    }

    default:
        return;

    }
}

}
