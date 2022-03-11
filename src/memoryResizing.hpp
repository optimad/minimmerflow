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

#ifndef __MINIMMERFLOW_ADAPTATIONMANAGER_HPP__
#define __MINIMMERFLOW_ADAPTATIONMANAGER_HPP__

#include <utils_cuda.hpp>

#include <bitpit_containers.hpp>

#include <vector>
#include <array>
#include <cuda.h>

class MemoryResizing
{
public:

     MemoryResizing();
    ~MemoryResizing();

    template<typename T>
    void cuda_setPtr(T **ptr);

    CUresult cuda_grow(std::size_t new_sz);

protected:

    void cuda_setReservedSize(const size_t reservedSize);

private:

    CUdeviceptr *m_dp;
    CUmemAllocationProp m_prop;
    CUmemAccessDesc m_accessDesc;
    struct Range {
        CUdeviceptr start;
        size_t sz;
    };
    std::vector<Range> m_vaRanges;
    std::vector<CUmemGenericAllocationHandle> m_handles;
    std::vector<size_t> m_handleSizes;
    size_t m_allocSize;
    size_t m_reservedSize;
    size_t m_chunkSize;

    CUresult cuda_reserve(size_t new_sz);
    void cuda_addVARange(CUdeviceptr new_ptr, size_t new_range);

};

    #include "memoryResizing.tpp"

#endif
