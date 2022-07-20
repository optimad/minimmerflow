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

#ifndef __MINIMMERFLOW_TEST_HPP__
#define __MINIMMERFLOW_TEST_HPP__

#include "containers.hpp"

namespace test {


void plotContainer(ScalarStorage<std::size_t> &container, std::size_t size);
void plotContainerCollection(ScalarStorageCollection<std::size_t> &container, std::size_t size);
void plotPiercedStorage(ScalarPiercedStorage<std::size_t> &container, std::size_t size);
void plotPiercedStorageCollection(ScalarPiercedStorageCollection<std::size_t> &container, std::size_t size);

void plotContainer(ScalarStorage<int> &container, std::size_t size);
void plotContainerCollection(ScalarStorageCollection<int> &container, std::size_t size);
void plotPiercedStorage(ScalarPiercedStorage<int> &container, std::size_t size);
void plotPiercedStorageCollection(ScalarPiercedStorageCollection<int> &container, std::size_t size);

#if ENABLE_CUDA
void cuda_plotContainer(ScalarStorage<std::size_t> &container, std::size_t size);
void cuda_plotContainerCollection(ScalarStorageCollection<std::size_t> &container, std::size_t size);
void cuda_plotPiercedStorage(ScalarPiercedStorage<std::size_t> &container, std::size_t size);
void cuda_plotPiercedStorageCollection(ScalarPiercedStorageCollection<std::size_t> &container, std::size_t size);

void cuda_plotContainer(ScalarStorage<int> &container, std::size_t size);
void cuda_plotContainerCollection(ScalarStorageCollection<int> &container, std::size_t size);
void cuda_plotPiercedStorage(ScalarPiercedStorage<int> &container, std::size_t size);
void cuda_plotPiercedStorageCollection(ScalarPiercedStorageCollection<int> &container, std::size_t size);
#endif

}

#endif
