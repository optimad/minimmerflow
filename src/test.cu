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
//      printf(" TEST pointer0 %ld ", deviceData[0][i]);
//      printf(" TEST pointer1 %ld ", deviceData[1][i]);
//      printf(" TEST pointer2 %ld ", deviceData[2][i]);
//      printf(" TEST pointer3 %ld ", deviceData[3][i]);
//      printf(" TEST pointer4 %ld ", deviceData[4][i]);
        deviceData[0][i] += 1;
        deviceData[1][i] += 1;
        deviceData[2][i] += 1;
        deviceData[3][i] += 1;
        deviceData[4][i] += 1;
    }
}


__global__ void dev_plotScalarStorage(std::size_t *deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
        deviceData[i] += 1;
    }
}


void cuda_plotContainer(ScalarStorage<std::size_t> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainer<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotContainerCollection(ScalarStorageCollection<std::size_t> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}

void cuda_plotPiercedStorage(ScalarPiercedStorage<std::size_t> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotScalarStorage<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotPiercedStorageCollection(ScalarPiercedStorageCollection<std::size_t> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}


__global__ void dev_plotContainer(int *deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
//      printf(" TEST container[%ld] = %ld\n ", i, deviceData[i]);
        deviceData[i] += 1;
    }
}


__global__ void dev_plotContainerCollection(int **deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
//      printf(" TEST pointer0 %ld ", deviceData[0][i]);
//      printf(" TEST pointer1 %ld ", deviceData[1][i]);
//      printf(" TEST pointer2 %ld ", deviceData[2][i]);
//      printf(" TEST pointer3 %ld ", deviceData[3][i]);
//      printf(" TEST pointer4 %ld ", deviceData[4][i]);
        deviceData[0][i] += 1;
        deviceData[1][i] += 1;
        deviceData[2][i] += 1;
        deviceData[3][i] += 1;
        deviceData[4][i] += 1;
    }
}


__global__ void dev_plotScalarStorage(int *deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
        deviceData[i] += 1;
    }
}


void cuda_plotContainer(ScalarStorage<int> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainer<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotContainerCollection(ScalarStorageCollection<int> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}

void cuda_plotPiercedStorage(ScalarPiercedStorage<int> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotScalarStorage<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotPiercedStorageCollection(ScalarPiercedStorageCollection<int> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}


__global__ void dev_plotContainer(double *deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
        deviceData[i] += 1;
    }
}


__global__ void dev_plotContainerCollection(double **deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
        deviceData[0][i] += 1;
        deviceData[1][i] += 1;
        deviceData[2][i] += 1;
        deviceData[3][i] += 1;
        deviceData[4][i] += 1;
    }
}


__global__ void dev_plotScalarStorage(double *deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
        deviceData[i] += 1;
    }
}


void cuda_plotContainer(ScalarStorage<double> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainer<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotContainerCollection(ScalarStorageCollection<double> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}

void cuda_plotPiercedStorage(ScalarPiercedStorage<double> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotScalarStorage<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotPiercedStorageCollection(ScalarPiercedStorageCollection<double> &container, std::size_t size)
{
    int numThreads = 256;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}

}
