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

#include "body.hpp"
#include "computation_info.hpp"
#include "constants.hpp"
#include "euler.hpp"
#include "problem.hpp"
#include "reconstruction.hpp"
#include "solver_writer.hpp"
#include "storage.hpp"
#include "utils.hpp"
#include "memory.hpp"
#include "adaptation.hpp"

#include <bitpit_IO.hpp>
#include <bitpit_voloctree.hpp>

#if ENABLE_MPI
#include <mpi.h>
#endif
#include <vector>
#include <time.h>
#if ENABLE_CUDA
#include <cuda.h>
#endif

#include "test.hpp"

using namespace bitpit;

void test1()
{
    ScalarStorage<std::size_t> simpleContainer;
    std::size_t initSize = 1024;
    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        simpleContainer.push_back(1);
    }
    simpleContainer.cuda_allocateDevice();
    simpleContainer.cuda_updateDevice();

    test::plotContainer(simpleContainer, simpleContainer.cuda_deviceDataSize());


    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        simpleContainer[iter] = 0;
    }
    simpleContainer.cuda_updateHost();
    std::size_t sum = 0;
    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        if (simpleContainer[iter] != 2) std::cout <<  "problem1 simpleContainer[" << iter << "] " << simpleContainer[iter] << std::endl;
        sum += simpleContainer[iter];
    }

    std::size_t valSum = 2 * initSize * initSize;

    std::cout << "Non-resized containers: sum = " << sum
              << " and valSum = " << valSum
              << std::endl;


    simpleContainer.clear();
    simpleContainer.resize(0);
    std::size_t newSize = 4*initSize;
    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        simpleContainer.push_back(2);
    }
    simpleContainer.cuda_resize(simpleContainer.cuda_deviceDataSize());
    simpleContainer.cuda_updateDevice();
    test::plotContainer(simpleContainer, simpleContainer.cuda_deviceDataSize());


    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        simpleContainer[iter] = 0;
    }
    simpleContainer.cuda_updateHost();

    sum = 0;
    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        if (simpleContainer[iter] != 3) std::cout <<  "problem2 simpleContainer[" << iter << "] " << simpleContainer[iter] << std::endl;
        sum += simpleContainer[iter];
    }
    valSum = 3 * newSize * newSize;
    std::cout << "Resized containers: sum = " << sum
              << " and valSum = " << valSum
              << std::endl;
    if (sum == valSum) {
        std::cout << "\nTEST #1: SUCCESSFULL" << std::endl;
    } else {
        std::cout << "\nTEST #1: UNSUCCESSFULL" << std::endl;
    }

}


void test2()
{
    ScalarStorageCollection<std::size_t> collectionContainer(N_FIELDS);
    std::size_t initSize = 1024;
    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        for (std::size_t iF = 0; iF < N_FIELDS; iF++) {
            collectionContainer[iF].push_back(iF);
        }
    }
    collectionContainer.cuda_allocateDevice();
    collectionContainer.cuda_updateDevice();

    test::plotContainerCollection(collectionContainer, initSize*initSize);

    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        for (int iF = 0; iF < N_FIELDS; iF++) {
            collectionContainer[iF][iter] = 0;
        }
    }
    collectionContainer.cuda_updateHost();
    std::vector<std::size_t> sum(N_FIELDS, 0);
    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        for (int iF = 0; iF < N_FIELDS; iF++) {
            sum[iF] += collectionContainer[iF][iter];
        }
    }

    std::size_t valSum = initSize * initSize;

    for (int iF = 0; iF < N_FIELDS; iF++) {
        std::cout << "Field ID" << iF << " -- sum = " << sum[iF]
                  << " and valSum = " << valSum * (iF + 1) << std::endl;
    }


     // ------------------------------ //


    std::size_t newSize = 4*initSize;
    for (int iF = 0; iF < N_FIELDS; iF++) {
        collectionContainer[iF].resize(newSize * newSize);
    }
    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        for (std::size_t  iF = 0; iF < N_FIELDS; iF++) {
            collectionContainer[iF][iter] = iF;
        }
    }

    collectionContainer.cuda_resize(newSize * newSize);
    collectionContainer.cuda_updateDevice();
    test::plotContainerCollection(collectionContainer, newSize * newSize);


    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        for (int iF = 0; iF < N_FIELDS; iF++) {
            collectionContainer[iF][iter] = 0;
        }
    }
    collectionContainer.cuda_updateHost();

    sum = std::vector<std::size_t>(N_FIELDS, 0);
    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        for (int iF = 0; iF < N_FIELDS; iF++) {
            sum[iF] += collectionContainer[iF][iter];
        }
    }
    valSum = newSize * newSize;

    for (int iF = 0; iF < N_FIELDS; iF++) {
        std::cout << "Field ID" << iF << " -- sum = " << sum[iF]
                  << " and valSum = " << valSum * (iF + 1) << std::endl;
    }

    bool check  = true;
    for (int iF = 0; iF < N_FIELDS; iF++) {
        check == check && (sum[iF] == valSum * (iF + 1));
    }
    if (check) {
        std::cout << "\nTEST #2: SUCCESSFULL" << std::endl;
    } else {
        std::cout << "\nTEST #2: UNSUCCESSFULL" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // Get the primary context and bind to it
    CUcontext ctx;
    CUdevice dev;
    CUDA_DRIVER_ERROR_CHECK(cuInit(0)); // Initialize the driver API, before calling any other driver API function
    CUDA_DRIVER_ERROR_CHECK(cuDevicePrimaryCtxRetain(&ctx, 0));
    CUDA_DRIVER_ERROR_CHECK(cuCtxSetCurrent(ctx));
    CUDA_DRIVER_ERROR_CHECK(cuCtxGetDevice(&dev));


    //
    // Initialization
    //

#if ENABLE_MPI
    // MPI Initialization
    MPI_Init(&argc, &argv);
#endif

    // test1
    try{
        std::cout << "EXECUTING TEST #1" << std::endl;
        test1();
        std::cout << "\n" << std::endl;
    }
    catch(std::exception & e){
        std::cout << "TEST #1 exited with an error of type : " << e.what() << std::endl;
        return 1;
    }

    // test2
    try{
        std::cout << "EXECUTING TEST #2" << std::endl;
        test2();
        std::cout << "\n" << std::endl;
    }
    catch(std::exception & e){
        std::cout << "TEST #2 exited with an error of type : " << e.what() << std::endl;
        return 1;
    }

    //
    // Finalization
    //

#if ENABLE_MPI
    // MPI finalization
    MPI_Finalize();
#endif
}
