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

#include "constants.hcu"
#include "utils.hcu"

namespace utils {

/*!
 * Calculates normal velocity from primitive variables.
 *
 * \param fields are the fields
 * \param n is the normal vector
 * \result The normal velocity.
 */
__device__ double dev_normalVelocity(const double *fields, const double *n)
{
    return (fields[DEV_FID_U] * n[0] + fields[DEV_FID_V] * n[1] + fields[DEV_FID_W] * n[2]);
}

/*!
 * Compute primitive variables from conservative variables for a ideal gas.
 *
 * \param c are the conservative variables
 * \param[out] p are the primitive variables
 */
__device__ void dev_conservative2primitive(const double *c, double *p)
{
    // Kinetic energy
    double K = (c[DEV_FID_RHO_U]*c[DEV_FID_RHO_U] + c[DEV_FID_RHO_V]*c[DEV_FID_RHO_V] + c[DEV_FID_RHO_W]*c[DEV_FID_RHO_W])/(c[DEV_FID_RHO]*c[DEV_FID_RHO]);

    // Temperature
    p[DEV_FID_T] = (2.0*c[DEV_FID_RHO_E]/c[DEV_FID_RHO] - K) / (2.0 / (DEV_GAMMA - 1.0));

    // Velocity
    p[DEV_FID_U] = c[DEV_FID_RHO_U] / c[DEV_FID_RHO];
    p[DEV_FID_V] = c[DEV_FID_RHO_V] / c[DEV_FID_RHO];
    p[DEV_FID_W] = c[DEV_FID_RHO_W] / c[DEV_FID_RHO];

    // Pressure
    p[DEV_FID_P] = c[DEV_FID_RHO] * p[DEV_FID_T];
}

/*!
 * Compute conservative variables from primitive variables for a ideal gas.
 *
 * \param p are the primitive variables
 * \param[out] c are the conservative variables
 */
__device__ void dev_primitive2conservative(const double *p, double *c)
{
    // Density
    c[DEV_FID_RHO] = p[DEV_FID_P] / p[DEV_FID_T];

    // Momentum
    c[DEV_FID_RHO_U] = c[DEV_FID_RHO] * p[DEV_FID_U];
    c[DEV_FID_RHO_V] = c[DEV_FID_RHO] * p[DEV_FID_V];
    c[DEV_FID_RHO_W] = c[DEV_FID_RHO] * p[DEV_FID_W];

    // Total energy
    c[DEV_FID_RHO_E] = c[DEV_FID_RHO] * p[DEV_FID_T] / (DEV_GAMMA - 1.0) + 0.5*c[DEV_FID_RHO]*(p[DEV_FID_U]*p[DEV_FID_U] + p[DEV_FID_V]*p[DEV_FID_V] + p[DEV_FID_W]*p[DEV_FID_W]);
}

}
