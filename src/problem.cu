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
#include "problem.hcu"
#include "problem.hpp"

namespace problem {

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
__device__ void dev_getBorderBCInfo(int problemType, int BCType, const double3 &point,
                                    const double3 &normal, DeviceProxyArray<double, 1> *info)
{
    BITPIT_UNUSED(point);
    BITPIT_UNUSED(normal);

    switch (problemType) {

    case (PROBLEM_FFSTEP):
    {
        switch (BCType) {

        case (BC_DIRICHLET):
            (*info)[DEV_FID_U] = 3.0;
            (*info)[DEV_FID_V] = 0.;
            (*info)[DEV_FID_W] = 0.;
            (*info)[DEV_FID_P] = 1.;
            (*info)[DEV_FID_T] = 1./1.4;
            return;

        }
    }

    default:
        return;

    }
}

}
