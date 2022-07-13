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
    printf("TEST size = %ld\n", size);
    if  (i < size)  {
        if (i == 0) printf(" TEST container");
        if (i == 0) printf(" TEST container0 = %p\n ",  deviceData[0]);
        if (i == 0) printf(" TEST container1 = %p\n ",  deviceData[1]);
        if (i == 0) printf(" TEST container2 = %p\n ",  deviceData[2]);
        if (i == 0) printf(" TEST container0[0] = %f\n ",  deviceData[0][0]);
        if (i == 0) printf(" TEST container1[0] = %f\n ",  deviceData[1][0]);
        if (i == 0) printf(" TEST container2[0] = %f\n ",  deviceData[2][0]);
        deviceData[0][i] += 1;
        deviceData[1][i] += 1;
        deviceData[2][i] += 1;
//      deviceData[3][i] += 1;
//      deviceData[4][i] += 1;
    }
}


__global__ void dev_plotScalarStorage(double *deviceData, std::size_t size)
{
    // Process interfaces
    std::size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if  (i < size)  {
        if (i == 0) printf(" TEST container = %p\n ",  deviceData);
        if (i == 0) printf(" TEST container[%ld] = %f\n ", i, deviceData[i]);
        deviceData[i] += 1;
    }
}


void cuda_plotContainer(ScalarStorage<double> &container, std::size_t size)
{
    int numThreads = 4;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainer<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotContainerCollection(ScalarStorageCollection<double> &container, std::size_t size)
{
    int numThreads = 4;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}

void cuda_plotPiercedStorage(ScalarPiercedStorage<double> &container, std::size_t size)
{
    int numThreads = 4;
    int numBlocks = ((size + numThreads - 1) / numThreads);

    std::cout << "CUDA container.cuda_deviceData() " << container.cuda_deviceData() << std::endl;
    dev_plotScalarStorage<<<numBlocks,numThreads>>>(container.cuda_deviceData(), size);
}

void cuda_plotPiercedStorageCollection(ScalarPiercedStorageCollection<double> &container, std::size_t size)
{
    int numThreads = 4;
    int numBlocks = ((size + numThreads - 1) / numThreads);


//  std::cout << "container.cuda_deviceCollectionData()[0]  " << container.cuda_deviceCollectionData()[0] << std::endl;
//  std::cout << "container.cuda_deviceCollectionData()[1]  " << container.cuda_deviceCollectionData()[1] << std::endl;
//  std::cout << "container.cuda_deviceCollectionData()[2]  " << container.cuda_deviceCollectionData()[2] << std::endl;
    std::cout << "size:  " << size << std::endl;
    std::cout << "container.cuda_deviceCollectionData()+0:  " << container.cuda_deviceCollectionData()+0 << std::endl;
    std::cout << "container.cuda_deviceCollectionData()+1:  " << container.cuda_deviceCollectionData()+1 << std::endl;
    std::cout << "container.cuda_deviceCollectionData()+2:  " << container.cuda_deviceCollectionData()+2 << std::endl;
    std::cout << "container[0].cuda_deviceData()  " << container[0].cuda_deviceData() << std::endl;
    std::cout << "container[1].cuda_deviceData()  " << container[1].cuda_deviceData() << std::endl;
    std::cout << "container[2].cuda_deviceData()  " << container[2].cuda_deviceData() << std::endl;
    dev_plotContainerCollection<<<numBlocks,numThreads>>>(container.cuda_deviceCollectionData(), size);
}

}
