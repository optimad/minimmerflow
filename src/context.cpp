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

#include "context.hpp"
#include <cuda.h>
#include "utils_cuda.hpp"

namespace context {
/*
 * Check of whether primary context exists, otherwise create it
 * \param[out] return the primary context
*/
CUcontext getPrimaryContext()
{
    CUcontext context{0};
    CUDA_DRIVER_ERROR_CHECK(cuCtxGetCurrent(&context));
    CUdevice device;
    CUDA_DRIVER_ERROR_CHECK(cuCtxGetDevice(&device));
    CUcontext primary_context;
    CUDA_DRIVER_ERROR_CHECK(cuDevicePrimaryCtxRetain(&primary_context, device));
    unsigned int flags;
    int active;
    CUresult primary_state_check_result =
        cuDevicePrimaryCtxGetState(device, &flags, &active);

    bool unInitializedContext =
        (primary_state_check_result == CUDA_ERROR_DEINITIALIZED)
     || (primary_state_check_result == CUDA_ERROR_NOT_INITIALIZED);

    // Check whether the primary context exists and if not, retain it
    if (unInitializedContext)
        CUDA_DRIVER_ERROR_CHECK(cuDevicePrimaryCtxRetain(&primary_context, device));

    // Check whether current context is the primary context
    if (context != primary_context) {
        return primary_context;
    }
    return context;
}


}
