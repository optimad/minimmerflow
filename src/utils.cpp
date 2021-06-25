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

# include "constants.hpp"
# include "utils.hpp"

using namespace bitpit;

namespace utils {

/*!
 * Calculates normal velocity from primitive variables.
 *
 * \param fields are the fields
 * \param n is the normal vector
 * \result The normal velocity.
 */
double normalVelocity(const double *fields, const std::array<double, 3> &n)
{
    return (fields[FID_U] * n[0] + fields[FID_V] * n[1] + fields[FID_W] * n[2]);
}

/*!
 * Compute primitive variables from conservative variables for a ideal gas.
 *
 * \param c are the conservative variables
 * \param[out] p are the primitive variables
 */
void conservative2primitive(const double *c, double *p)
{
    // Kinetic energy
    double K = (c[FID_RHO_U]*c[FID_RHO_U] + c[FID_RHO_V]*c[FID_RHO_V] + c[FID_RHO_W]*c[FID_RHO_W])/(c[FID_RHO]*c[FID_RHO]);

    // Temperature
    p[FID_T] = (2.0*c[FID_RHO_E]/c[FID_RHO] - K) / (2.0 / (GAMMA - 1.0));

    // Velocity
    p[FID_U] = c[FID_RHO_U] / c[FID_RHO];
    p[FID_V] = c[FID_RHO_V] / c[FID_RHO];
    p[FID_W] = c[FID_RHO_W] / c[FID_RHO];

    // Pressure
    p[FID_P] = c[FID_RHO] * p[FID_T];
}

/*!
 * Compute conservative variables from primitive variables for a ideal gas.
 *
 * \param p are the primitive variables
 * \param[out] c are the conservative variables
 */
void primitive2conservative(const double *p, double *c)
{
    // Density
    c[FID_RHO] = p[FID_P] / p[FID_T];

    // Momentum
    c[FID_RHO_U] = c[FID_RHO] * p[FID_U];
    c[FID_RHO_V] = c[FID_RHO] * p[FID_V];
    c[FID_RHO_W] = c[FID_RHO] * p[FID_W];

    // Total energy
    c[FID_RHO_E] = c[FID_RHO] * p[FID_T] / (GAMMA - 1.0) + 0.5*c[FID_RHO]*(p[FID_U]*p[FID_U] + p[FID_V]*p[FID_V] + p[FID_W]*p[FID_W]);
}

}
