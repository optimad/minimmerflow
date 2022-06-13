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

#include "memoryResizing.hpp"

#include "context.hpp"
#include <cuda.h>

/*!
 * Constructor
 */
MemoryResizing::MemoryResizing()
:
    m_dp(0Ull), m_prop(), m_handles(), m_allocSize(0ULL), m_reservedSize(0ULL), m_chunkSize(0ULL)
{
    CUresult status = CUDA_SUCCESS;

    // Get the current context
    CUcontext curr_ctx;
//  CUDA_DRIVER_ERROR_CHECK(cuCtxGetCurrent(&curr_ctx));
//  assert(status == CUDA_SUCCESS); // Blocking if status is not CUDA_SUCCESS

//  // If there is no context, use the primary context
//  if (curr_ctx == nullptr)
//  {
        // This is a safe way to get the primary context:
        curr_ctx = context::getPrimaryContext();
        status = cuCtxSetCurrent(curr_ctx);
        CUDA_DRIVER_ERROR_CHECK(status);
//  }

    // Get the device
    CUdevice device;
    CUDA_DRIVER_ERROR_CHECK(cuCtxGetDevice(&device));

    m_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    m_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    m_prop.location.id = (int)device;

    m_accessDesc.location = m_prop.location;
    m_accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUDA_DRIVER_ERROR_CHECK(cuMemGetAllocationGranularity(&m_chunkSize, &m_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    assert(status == CUDA_SUCCESS);
}


/*!
 * Destructor
 */
MemoryResizing::~MemoryResizing()
{
    CUresult status = CUDA_SUCCESS;
    (void)status;
    if (m_dp != 0ULL) {
        status = cuMemUnmap(m_dp, m_allocSize);
        CUDA_DRIVER_ERROR_CHECK(status);
        assert(status == CUDA_SUCCESS);
        for (size_t i = 0; i < m_vaRanges.size(); i++) {
            status = cuMemAddressFree(m_vaRanges[i].start, m_vaRanges[i].sz);
            CUDA_DRIVER_ERROR_CHECK(status);
            assert(status == CUDA_SUCCESS);
        }
        for (size_t i = 0ULL; i < m_handles.size(); i++) {
            status = cuMemRelease(m_handles[i]);
            CUDA_DRIVER_ERROR_CHECK(status);
            assert(status == CUDA_SUCCESS);
        }
    }
}



/*!
 * Reserve virtual memory of array on GPU
 */
CUresult MemoryResizing::cuda_reserve(size_t new_sz)
{
    CUresult status = CUDA_SUCCESS;

    CUdeviceptr new_ptr = 0ULL;

    if (new_sz <= m_reservedSize) {
        return CUDA_SUCCESS;
    }

    const size_t aligned_sz = ((new_sz + m_chunkSize - 1) / m_chunkSize) * m_chunkSize;

    status = cuMemAddressReserve(&new_ptr, (aligned_sz - m_reservedSize), 0ULL, m_dp + m_reservedSize, 0ULL);

    // Try to reserve an address just after what we already have reserved
    if (status != CUDA_SUCCESS || (new_ptr != m_dp + m_reservedSize)) {
        if (new_ptr != 0ULL) {
            (void)cuMemAddressFree(new_ptr, (aligned_sz - m_reservedSize));
        }
        // Slow path - try to find a new address reservation big enough for us
        status = cuMemAddressReserve(&new_ptr, aligned_sz, 0ULL, 0U, 0);
        if (status == CUDA_SUCCESS && m_dp != 0ULL) {
            CUdeviceptr ptr = new_ptr;
            // Found one, now unmap our previous allocations
            status = cuMemUnmap(m_dp, m_allocSize);
            CUDA_DRIVER_ERROR_CHECK(status);
            for (size_t i = 0ULL; i < m_handles.size(); i++) {
                const size_t hdl_sz = m_handleSizes[i];
                // And remap them, enabling their access
                status = cuMemMap(ptr, hdl_sz, 0ULL, m_handles[i], 0ULL);
                if (status != CUDA_SUCCESS)
                    break;
                status = cuMemSetAccess(ptr, hdl_sz, &m_accessDesc, 1ULL);
                if (status != CUDA_SUCCESS)
                    break;
                ptr += hdl_sz;
            }
            if (status != CUDA_SUCCESS) {
                // Failed the mapping somehow... clean up!
                status = cuMemUnmap(new_ptr, aligned_sz);
                CUDA_DRIVER_ERROR_CHECK(status);
                status = cuMemAddressFree(new_ptr, aligned_sz);
                CUDA_DRIVER_ERROR_CHECK(status);
            }
            else {
                // Clean up our old VA reservations!
                for (size_t i = 0ULL; i < m_vaRanges.size(); i++) {
                    (void)cuMemAddressFree(m_vaRanges[i].start, m_vaRanges[i].sz);
                }
                m_vaRanges.clear();
            }
        }
        // Assuming everything went well, update everything
        if (status == CUDA_SUCCESS) {
            m_dp = new_ptr;
            m_reservedSize = aligned_sz;
            cuda_addVARange(new_ptr, aligned_sz);
        }
    }
    else {
        cuda_addVARange(new_ptr, aligned_sz - m_reservedSize);
        if (m_dp == 0ULL) {
            m_dp = new_ptr;
        }
        m_reservedSize = aligned_sz;
    }

    return status;
}


/*!
 * Grow array on GPU
 */
CUresult MemoryResizing::cuda_grow(std::size_t new_sz)
{
    CUresult status = CUDA_SUCCESS;
    CUmemGenericAllocationHandle handle;
    if (new_sz <= m_allocSize) {
        return CUDA_SUCCESS;
    }

    const size_t size_diff = new_sz - m_allocSize;
    // Round up to the next chunk size
    const size_t sz = ((size_diff + m_chunkSize - 1) / m_chunkSize) * m_chunkSize;
    status = cuda_reserve(m_allocSize + sz);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    status = cuMemCreate(&handle, sz, &m_prop, 0);
    if (status == CUDA_SUCCESS) {
        status = cuMemMap(m_dp + m_allocSize, sz, 0ULL, handle, 0ULL);
        if (status == CUDA_SUCCESS) {
            status = cuMemSetAccess(m_dp + m_allocSize, sz, &m_accessDesc, 1ULL);
            if (status == CUDA_SUCCESS) {
                cuda_addHandleInfo(handle, sz);
            }
            if (status != CUDA_SUCCESS) {
                (void)cuMemUnmap(m_dp + m_allocSize, sz);
            }
        }
        if (status != CUDA_SUCCESS) {
            (void)cuMemRelease(handle);
        }
    }
    return status;
}

/*!
 * Get the CUdeviceptr
 */
CUdeviceptr MemoryResizing::getCUdeviceptr()
{
    return m_dp;
}


/*!
 * Set the allocated size
 * \param[in] the allocated size
 */
void MemoryResizing::cuda_setReservedSize(const size_t reservedSize)
{
    m_allocSize = reservedSize;
}


/*!
 * Add a new Range to m_vaRanges
 * \param[in] new_ptr the CUdeviceptr
 * \param[in] new_rangeSz the size of the new range
 */
void MemoryResizing::cuda_addVARange(CUdeviceptr new_ptr, size_t new_rangeSz)
{
    Range r;
    r.start = new_ptr;
    r.sz = new_rangeSz;
    m_vaRanges.push_back(r);
}

/*!
 * Store info about new handle
 * \param[in] handle the CUmemGenericAllocationHandle
 * \param[in] sz the size of the memory handled by the handle
 */
void MemoryResizing::cuda_addHandleInfo(CUmemGenericAllocationHandle &handle, size_t sz)
{
    m_handles.push_back(handle);
    m_handleSizes.push_back(sz);
    m_allocSize += sz;
}
