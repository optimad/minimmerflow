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

//uncomment to show VERBOSE information at runtime
#define _WITH_DEBUG_VERBOSE

#include "memoryResizing.hpp"

#include "context.hpp"
#include <cassert>
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
//  CUDA_DRIVER_ERROR_CHECK(MemoryResizing::cuda_free());
}

/*!
 * Free mechanism
 */
CUresult MemoryResizing::cuda_free()
{
    CUresult status = CUDA_SUCCESS;
    if (m_dp != 0ULL) {
        CUdeviceptr ptr = m_dp;
        for (size_t i = 0ULL; i < m_handles.size(); i++) {
          const size_t hdl_sz = m_handleSizes[i];
          // And remap them, enabling their access
          status = cuMemUnmap(ptr, hdl_sz);
          CUDA_DRIVER_ERROR_CHECK(status);
          ptr += hdl_sz;
        }
        for (size_t i = 0ULL; i < m_handles.size(); i++) {
            status = cuMemRelease(m_handles[i]);
            CUDA_DRIVER_ERROR_CHECK(status);
            assert(status == CUDA_SUCCESS);
        }
        assert(status == CUDA_SUCCESS);
        for (size_t i = 0; i < m_vaRanges.size(); i++) {
            status = cuMemAddressFree(m_vaRanges[i].start, m_vaRanges[i].sz);
            CUDA_DRIVER_ERROR_CHECK(status);
            assert(status == CUDA_SUCCESS);
        }

        m_allocSize = 0;
        m_reservedSize = 0;
        m_vaRanges.clear();
        m_handles.clear();
        m_handleSizes.clear();
        m_dp = 0ULL;
    }
    return status;
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
#ifdef _WITH_DEBUG_VERBOSE
    if (status == CUDA_SUCCESS) m_fastPath = true;
    std::cout << "m_fastPath " << m_fastPath
              << ", new_ptr " << new_ptr
              << ", m_dp " << m_dp
              << ", aligned_sz " << aligned_sz
              << ", m_reservedSize " << m_reservedSize << std::endl;
#endif

    // Try to reserve an address just after what we already have reserved
    if (status != CUDA_SUCCESS || (new_ptr != m_dp + m_reservedSize)) {
        if (new_ptr != 0ULL) {
            (void)cuMemAddressFree(new_ptr, (aligned_sz - m_reservedSize));
        }
        // Slow path - try to find a new address reservation big enough for us
        status = cuMemAddressReserve(&new_ptr, aligned_sz, 0ULL, 0U, 0);
        if (status == CUDA_SUCCESS) m_slowPath = true;
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
#ifdef _WITH_DEBUG_VERBOSE
    if (status == CUDA_SUCCESS)
        std::cerr << "# cuda_reserve extended previous pointer successfully"  << std::endl;
#endif // _WITH_DEBUG_VERBOSE

    return status;
}


/*!
 * Grow array on GPU
 */
CUresult MemoryResizing::cuda_grow(std::size_t new_sz)
{
    std::cout << "\n\n BEGIN OF GROW" << std::endl;
    m_ready2Grow = false;
    m_fastPath   = false;
    m_slowPath   = false;
    m_memCreate  = false;
    m_memMap     = false;
    m_memAccess  = false;

    CUresult status = CUDA_SUCCESS;
    CUmemGenericAllocationHandle handle;
    if (new_sz <= m_allocSize) {
#ifdef _WITH_DEBUG_VERBOSE
        std::cerr << "# cuda_grow ( requested = " << new_sz << ", reserved = " << m_allocSize << ") => no grow required\n";
#endif // _WITH_DEBUG_VERBOSE
        return CUDA_SUCCESS;
    }

    const size_t size_diff = new_sz - m_allocSize;
    // Round up to the next chunk size
    const size_t sz = ((size_diff + m_chunkSize - 1) / m_chunkSize) * m_chunkSize;
    status = cuda_reserve(m_allocSize + sz);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    m_ready2Grow = true;
    status = cuMemCreate(&handle, sz, &m_prop, 0);
    if (status == CUDA_SUCCESS) {
        m_memCreate = true;
        status = cuMemMap(m_dp + m_allocSize, sz, 0ULL, handle, 0ULL);
        if (status == CUDA_SUCCESS) {
            m_memMap = true;
            status = cuMemSetAccess(m_dp + m_allocSize, sz, &m_accessDesc, 1ULL);
            if (status == CUDA_SUCCESS) {
                m_memAccess = true;
                cuda_addHandleInfo(handle, sz);
            }
            if (status != CUDA_SUCCESS) {
                (void)cuMemUnmap(m_dp + m_allocSize, sz);
            }
        }
        if (status != CUDA_SUCCESS) {
#ifdef _WITH_DEBUG_VERBOSE
            std::cerr << "# cuda_grow, cuMemRelease in progress...\n";
#endif // _WITH_DEBUG_VERBOSE
            (void)cuMemRelease(handle);
        }
    }
#ifdef _WITH_DEBUG_VERBOSE
    cuda_debugInfo();
#endif // _WITH_DEBUG_VERBOSE
    std::cout << "\n\n END OF GROW" << std::endl;
    return status;
}


/*!
 * Get the CUdeviceptr
 */
CUdeviceptr MemoryResizing::getCUdeviceptr() const
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

size_t MemoryResizing::totalMemSize() const
{
  size_t m;
  CUdevice device;
  cuCtxGetDevice(&device);
  cuDeviceTotalMem(&m, device);
  return m;
}

/*!
 * Write class status to ostream
 * \param[in] handle the MemoryResizing
 */
std::ostream& operator<< (std::ostream& out, const MemoryResizing& mr)
{
  out << "[MemoryResizing : "
      << " address: " << mr.getCUdeviceptr()
      << " allocated: " << mr.allocSize()
      << " reserved: " << mr.reservedSize()
      << " chunk size: " << mr.chunkSize()
  ;

  out << " Range [ ";
  for (auto& r: mr.m_vaRanges) {
    out << "(" << r.start << ", " << r.sz << ") ";
  }
  out << "]";

  out << " Handles [ ";
  for (auto& r: mr.m_handleSizes) {
    out << r << " ";
  }
  out << "]";

  return out;
}

void MemoryResizing::cuda_debugInfo() {

    std::cerr << "\n\n# cuda_grow, BEGIN OF DEBUGINFO \n"<< std::endl;

    if (m_ready2Grow) {
        std::cerr << "# cuda_grow, reserved successfully and ready to grow \n";
    } else {
        std::cerr << "# WARNING in cuda_grow, reserved was unsuccessful \n";
    }

    if (m_fastPath) {
        std::cerr << "# cuda_reserve, fast path followed\n";
    }

    if (m_slowPath) {
        std::cerr << "# cuda_reserve, start to follow the slow path\n";
    }

    if ((m_fastPath == false) && (m_slowPath == false)) {
        std::cerr << "# WARNING in cuda_reserve, neither fast or slow path\n";
    }

    if (m_memCreate) {
        std::cerr << "# cuda_grow, cuMemCreate was successful \n";
    } else {
        std::cerr << "# WARNING in cuda_grow, cuMemCreate was unsuccessful \n";
    }

    if (m_memMap) {
        std::cerr << "# cuda_grow, cuMemMap was successful \n";
    } else {
        std::cerr << "# WARNING in cuda_grow, cuMemMap was unsuccessful \n";
    }

    if (m_memAccess) {
        std::cerr << "# cuda_grow, cuMemSetAccess was successful \n";
    } else {
        std::cerr << "# WARNING in cuda_grow, cuMemSetAccess was unsuccessful \n";
    }

//  cuda_debugStats();
    std::cerr << "\n# cuda_grow, END OF DEBUGINFO \n\n"<< std::endl;
}


void MemoryResizing::cuda_debugStats() {

    std::cerr << "[MemoryResizing : "
        << " address: " << this->getCUdeviceptr()
        << " allocated: " << this->allocSize()
        << " reserved: " << this->reservedSize()
        << " chunk size: " << this->chunkSize();

    std::cerr << " Range [ ";
    for (auto& r: m_vaRanges) {
        std::cout << "(" << r.start << ", " << r.sz << ") ";
    }
    std::cerr << "]";

    std::cerr << " Handles [ ";
    for (auto& r: m_handleSizes) {
        std::cerr << r << " ";
    }
    std::cerr << "]";

}
