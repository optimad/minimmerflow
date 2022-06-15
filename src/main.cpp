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

    ScalarStorageCollection<std::size_t> cellRHS(N_FIELDS);
    std::size_t initSize = 1024;
    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        for (std::size_t iF = 0; iF < N_FIELDS; iF++) {
            cellRHS[iF].push_back(iF);
        }
    }
    cellRHS.cuda_allocateDevice();
    cellRHS.cuda_updateDevice();

    test::plotContainerCollection(cellRHS, initSize*initSize);

    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        for (int iF = 0; iF < N_FIELDS; iF++) {
            cellRHS[iF][iter] = 0;
        }
    }
    cellRHS.cuda_updateHost();
    std::vector<std::size_t> sum(N_FIELDS, 0);
    for (std::size_t iter = 0; iter < initSize*initSize; iter++) {
        for (int iF = 0; iF < N_FIELDS; iF++) {
            sum[iF] += cellRHS[iF][iter];
        }
    }

    double valSum = initSize * initSize;

    for (int iF = 0; iF < N_FIELDS; iF++) {
        std::cout << "iF " << iF << " -- sum = " << sum[iF]
                  << " and valSum = " << valSum * (iF + 1) << std::endl;
    }


     // ------------------------------ //


    std::size_t newSize = 4*initSize;
    for (int iF = 0; iF < N_FIELDS; iF++) {
        cellRHS[iF].resize(newSize * newSize);
    }
    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        for (std::size_t  iF = 0; iF < N_FIELDS; iF++) {
            cellRHS[iF][iter] = iF;
        }
    }

    cellRHS.cuda_resize(newSize * newSize);
    cellRHS.cuda_updateDevice();
    test::plotContainerCollection(cellRHS, newSize * newSize);


    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        for (int iF = 0; iF < N_FIELDS; iF++) {
            cellRHS[iF][iter] = 0;
        }
    }
    cellRHS.cuda_updateHost();

    sum = std::vector<std::size_t>(N_FIELDS, 0);
    for (std::size_t iter = 0; iter < newSize*newSize; iter++) {
        for (int iF = 0; iF < N_FIELDS; iF++) {
            sum[iF] += cellRHS[iF][iter];
        }
    }
    valSum = newSize * newSize;

    for (int iF = 0; iF < N_FIELDS; iF++) {
        std::cout << "iF " << iF << " -- sum = " << sum[iF] << " and valSum = " << valSum * (iF + 1) << std::endl;
    }

    //
    // Finalization
    //

#if ENABLE_MPI
    // MPI finalization
    MPI_Finalize();
#endif
}
