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


void MemoryResizing::cuda_setReservedSize(const size_t reservedSize)
{
    m_allocSize = reservedSize;
}


/*!
 * Add a new Range to m_vaRanges
 */
void MemoryResizing::cuda_addVARange(CUdeviceptr new_ptr, size_t new_rangeSz)
{
    Range r;
    r.start = new_ptr;
    r.sz = new_rangeSz;
    m_vaRanges.push_back(r);
}
