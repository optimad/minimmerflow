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
#include "containers.cu"
#include "utils.hcu"

namespace utils {

/*!
 * Calculates normal velocity from primitive variables.
 *
 * \param u is the x-component of the velocity vector
 * \param v is the y-component of the velocity vector
 * \param w is the z-component of the velocity vector
 * \param nx is the x-component of the normal vector
 * \param ny is the y-component of the normal vector
 * \param nz is the z-component of the normal vector
 * \result The normal velocity.
 */
__device__ double dev_normalVelocity(double u, double v, double w, double nx, double ny, double nz)
{
    return (u * nx + v * ny + w * nz);
}

/*!
 * Compute primitive variables from conservative variables for a ideal gas.
 *
 * \param c are the conservative variables
 * \param[out] p are the primitive variables
 */
__device__ void dev_conservative2primitive(const DeviceCollectionDataConstCursor<double> & __restrict__ c, double * __restrict__ p)
{
    // Density
    double rho = c[DEV_FID_RHO];

    double rho_inv  = 1. / rho;
    double rho2_inv = rho_inv * rho_inv;

    // Velocity
    double rho_u = c[DEV_FID_RHO_U];
    double rho_v = c[DEV_FID_RHO_V];
    double rho_w = c[DEV_FID_RHO_W];

    p[DEV_FID_U] = rho_u * rho_inv;
    p[DEV_FID_V] = rho_v * rho_inv;
    p[DEV_FID_W] = rho_w * rho_inv;

    double K = 0.5 * (rho_u * rho_u + rho_v * rho_v + rho_w * rho_w) * rho2_inv;

    // Temperature
    double rho_e = c[DEV_FID_RHO_E];

    double e = rho_e * rho_inv;
    double T = (e - K) * (DEV_GAMMA - 1.0);

    p[DEV_FID_T] = T;

    // Pressure
    p[DEV_FID_P] = rho * T;
}

/*!
 * Compute conservative variables from primitive variables for a ideal gas.
 *
 * \param p are the primitive variables
 * \param[out] c are the conservative variables
 */
__device__ void dev_primitive2conservative(const double * __restrict__ p, DeviceCollectionDataCursor<double> * __restrict__ c)
{
    // Continuity
    double T   = p[DEV_FID_T];
    double rho = p[DEV_FID_P] / T;

    (*c)[DEV_FID_RHO] = rho;

    // Momentum
    double u  = p[DEV_FID_U];
    double u2 = u * u;

    (*c)[DEV_FID_RHO_U] = rho * u;

    double v  = p[DEV_FID_V];
    double v2 = v * v;

    (*c)[DEV_FID_RHO_V] = rho * v;

    double w  = p[DEV_FID_W];
    double w2 = w * w;

    (*c)[DEV_FID_RHO_W] = rho * w;

    // Total energy
    double K = 0.5 * (u2 + v2 + w2);

    (*c)[DEV_FID_RHO_E] = rho * (T / (DEV_GAMMA - 1.0) + K);
}

}
