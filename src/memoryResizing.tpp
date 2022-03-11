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

#ifndef __MINIMMEFLOW_ADAPTATIONMANAGER_TPP__
#define __MINIMMEFLOW_ADAPTATIONMANAGER_TPP__


#include <cuda.h>

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
            << "failed (" << (unsigned)res << "): " << errStr << std::endl;
    }
}
#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

/*!
 * Set the CUdeviceptr
 */
template<typename T>
void MemoryResizing::cuda_setPtr(T **ptr)
{
    m_dp = (CUdeviceptr*) (uintptr_t*) ptr;
}


#endif
