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
#include "utils_cuda.hpp"

#include <iostream>

__constant__ int DEV_FID_P;
__constant__ int DEV_FID_U;
__constant__ int DEV_FID_V;
__constant__ int DEV_FID_W;
__constant__ int DEV_FID_T;

__constant__ int DEV_FID_RHO;
__constant__ int DEV_FID_RHO_U;
__constant__ int DEV_FID_RHO_V;
__constant__ int DEV_FID_RHO_W;
__constant__ int DEV_FID_RHO_E;

__constant__ int DEV_FID_EQ_C;
__constant__ int DEV_FID_EQ_M_X;
__constant__ int DEV_FID_EQ_M_Y;
__constant__ int DEV_FID_EQ_M_Z;
__constant__ int DEV_FID_EQ_E;

__constant__ float DEV_GAMMA;
__constant__ float DEV_R;

namespace constants
{

void cuda_initialize()
{
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_P, &FID_P, sizeof(FID_P)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_U, &FID_U, sizeof(FID_U)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_V, &FID_V, sizeof(FID_V)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_W, &FID_W, sizeof(FID_W)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_T, &FID_T, sizeof(FID_T)));

    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_RHO,   &FID_RHO,   sizeof(FID_RHO)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_RHO_U, &FID_RHO_U, sizeof(FID_RHO_U)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_RHO_V, &FID_RHO_V, sizeof(FID_RHO_V)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_RHO_W, &FID_RHO_W, sizeof(FID_RHO_W)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_RHO_E, &FID_RHO_E, sizeof(FID_RHO_E)));

    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_EQ_C,   &FID_EQ_C,   sizeof(FID_EQ_C)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_EQ_M_X, &FID_EQ_M_X, sizeof(FID_EQ_M_X)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_EQ_M_Y, &FID_EQ_M_Y, sizeof(FID_EQ_M_Y)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_EQ_M_Z, &FID_EQ_M_Z, sizeof(FID_EQ_M_Z)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_FID_EQ_E,   &FID_EQ_E,   sizeof(FID_EQ_E)));

    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_GAMMA, &GAMMA, sizeof(GAMMA)));
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(DEV_R,     &R,     sizeof(R)));
}

}
