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

#include <stdio.h>
#include "test.hcu"
#include "containers.hcu"

namespace test {


__global__ void dev_plotContainer(std::size_t *deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
//      printf(" TEST container[%ld] = %ld\n ", i, deviceData[i]);
        deviceData[i] += 1;
    }
}


__global__ void dev_plotContainerCollection(std::size_t **deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
//      printf(" TEST pointer0 %p ", deviceData[0]);
//      printf(" TEST pointer1 %p ", deviceData[1]);
//      printf(" TEST pointer2 %p ", deviceData[2]);
//      printf(" TEST pointer3 %p ", deviceData[3]);
//      printf(" TEST pointer4 %p ", deviceData[4]);
        deviceData[0][i] += 1;
        deviceData[1][i] += 1;
        deviceData[2][i] += 1;
        deviceData[3][i] += 1;
        deviceData[4][i] += 1;
    }
}


void cuda_plotContainer(ScalarStorage<std::size_t> &container, std::size_t size)
{
    int numThreads = 4;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainer<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotContainerCollection(ScalarStorageCollection<std::size_t> &container, std::size_t size)
{
    int numThreads = 4;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}

}
