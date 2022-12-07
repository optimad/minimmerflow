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

#include <cuda.h>
#include <iostream>

#ifndef __MINIMMERFLOW_UTILS_CUDA_HPP__
#define __MINIMMERFLOW_UTILS_CUDA_HPP__

#define CUDA_ERROR_CHECK(function_call)                                           \
{                                                                                 \
    auto errorCode = (function_call);                                             \
    if (errorCode != cudaSuccess) {                                               \
        std::cerr << "GPUassert: " << __FILE__ << ":" << __LINE__ << std::endl;   \
        std::cerr << "GPUassert: " << cudaGetErrorString(errorCode) << std::endl; \
	exit(-1);                                                                 \
    }                                                                            \
}

#define CUDA_DRIVER_ERROR_CHECK(call_res)                                                     \
{                                                                               \
    if (call_res != CUDA_SUCCESS) {                                             \
        const char *errStr = NULL;                                              \
        (void)cuGetErrorString(call_res, &errStr);                              \
        std::cerr << "Driver API Error: " << __FILE__ << ":" << __LINE__ << std::endl;   \
        std::cerr << "Driver API Error: "  << #call_res                         \
            << "failed (" << (unsigned)call_res << "): " << errStr << std::endl;\
	    exit(-1);                                                               \
    }                                                                           \
}

#endif

