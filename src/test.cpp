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

#include "test.hpp"

namespace test {

void plotContainer(ScalarStorage<std::size_t> &container, std::size_t size)
{
#if ENABLE_CUDA
    cuda_plotContainer(container, size);
#endif
}

void plotContainerCollection(ScalarStorageCollection<std::size_t> &container, std::size_t size)
{
#if ENABLE_CUDA
    cuda_plotContainerCollection(container, size);
#endif
}

void plotPiercedStorage(ScalarPiercedStorage<std::size_t> &container, std::size_t size)
{
#if ENABLE_CUDA
    cuda_plotPiercedStorage(container, size);
#endif
}

}
