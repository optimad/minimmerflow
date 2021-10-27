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

#include "reconstruction.hcu"

namespace reconstruction {

/*!
 * Evaluate the reconstruction are the specified point.
 *
 * \param order is the order
 * \param point is the point where the reconstruction will be evaluated
 * \param means are the mean values of the fields
 * \param[out] values on output will contain the reconstructed values
 */
__device__ void dev_eval(int order, const double *point, const double *means, double *values)
{
    switch (order) {

    case (1):
        dev_eval_1(point, means, values);
        break;

    default:
        break;

    }
}

/*!
 * Evaluate the reconstruction are the specified point.
 *
 * \param order is the order
 * \param point is the point where the reconstruction will be evaluated
 * \param means are the mean values of the fields
 * \param[out] values on output will contain the reconstructed values
 */
__device__ void dev_eval_1(const double *point, const double *means, double *values)
{
    BITPIT_UNUSED(point);

    for (int i = 0; i < N_FIELDS; ++i) {
        values[i] = means[i];
    }
}

}
