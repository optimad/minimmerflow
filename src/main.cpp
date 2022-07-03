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
#include "cuda_runtime.h"
#endif

#include "test.hpp"

using namespace bitpit;

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);


// Cuda-driver version of test 0
void test0DR() {

  MemoryResizing foo;

  std::cout << "TOTAL DEVICE MEMORY: " << foo.totalMemSize() << std::endl;
  std::cout << foo << std::endl;

  size_t chunk_bytes = 1024*1024;
  for (size_t requested = chunk_bytes; requested < foo.totalMemSize(); requested *= 2) {
    std::cout << "\n--- grow request (bytes) : " << requested << std::endl;
    foo.cuda_grow(requested);
    std::cout << foo << "\n" << std::endl;
  }

  foo.free();

  std::cout << foo << std::endl;
}


// Cuda-runtime version of test 0
void test0RT() {
  // trick to build primary context
  cudaSetDevice(0); cudaFree(0);

  test0DR();
}


// This test takes a ScalarStorage, resizes it, increments on GPU its elements
// and copies it back to CPU to validate the sum of its elements
void test1A()
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
        std::cout << "\nTEST #1A: SUCCESSFULL" << std::endl;
    } else {
        std::cout << "\nTEST #1A: UNSUCCESSFULL" << std::endl;
    }

}


// This test takes two ScalarStorages, resizes one of them, increments on GPU its elements
// and copies it back to CPU to validate the sum of its elements
void test1B()
{
    std::cout << "Ok 0" << std::endl;
    std::size_t initSize = 1000000000;
//  std::vector<std::size_t> v1(initSize);
    ScalarStorage<std::size_t> simpleContainer1(initSize);
    std::cout << "Ok 1" << std::endl;
//  std::vector<std::size_t> v2(initSize);
    ScalarStorage<std::size_t> simpleContainer2(initSize);

    std::cout << "Cuda Allocation" << std::endl;
    simpleContainer1.cuda_allocateDevice();
    simpleContainer2.cuda_allocateDevice();
    std::cout << "Cuda Update" << std::endl;
    simpleContainer1.cuda_updateDevice();
    simpleContainer2.cuda_updateDevice();
    simpleContainer1.cuda_fillDevice(1);
    test::plotContainer(simpleContainer1, simpleContainer1.cuda_deviceDataSize());
    simpleContainer1.cuda_updateHost();

    std::size_t sum = 0;
    for (std::size_t iter = 0; iter < initSize; iter++) {
        sum += simpleContainer1[iter];
    }
    std::size_t valSum = 2 * initSize;

    std::cout << "Resized containers: sum = " << sum
              << " and valSum = " << valSum
              << std::endl;
    std::cout << "Begin of CPU resizing" << std::endl;

    // Resize simpleContainer1
    simpleContainer1.clear();
    simpleContainer1.resize(0);
//  std::size_t newSize = 2*initSize;
    std::size_t newSize = 1010000000;
    for (std::size_t iter = 0; iter < newSize; iter++) {
        simpleContainer1.push_back(2);
    }
    std::cout << "End of CPU resizing" << std::endl;
    simpleContainer1.cuda_resize(simpleContainer1.cuda_deviceDataSize());
    simpleContainer1.cuda_updateDevice();
    test::plotContainer(simpleContainer1, simpleContainer1.cuda_deviceDataSize());


    for (std::size_t iter = 0; iter < newSize; iter++) {
        simpleContainer1[iter] = 0;
    }
    simpleContainer1.cuda_updateHost();

    sum = 0;
    for (std::size_t iter = 0; iter < newSize; iter++) {
        if (simpleContainer1[iter] != 3) std::cout <<  "problem2 simpleContainer1[" << iter << "] " << simpleContainer1[iter] << std::endl;
        sum += simpleContainer1[iter];
    }
    valSum = 3 * newSize;

    std::cout << "Resized containers: sum = " << sum
              << " and valSum = " << valSum
              << std::endl;
    if (sum == valSum) {
        std::cout << "\nTEST #1B: SUCCESSFULL" << std::endl;
    } else {
        std::cout << "\nTEST #1B: UNSUCCESSFULL" << std::endl;
    }

}

// This test takes a ScalarStorageCollection, resizes it, increments on GPU its
// elements and copies it back to CPU to validate the sum of its elements
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


// Resizes the CFD mesh, as needed for test3
void adaptMeshAndFields(ComputationInfo &computationInfo, VolOctree &mesh,
                        ScalarPiercedStorage<std::size_t> &cellFoo,
                        const problem::ProblemType problemType)
{

    adaptation::meshAdaptation(mesh);

#if ENABLE_CUDA
    // Resize CPU containers holding geometrical and computations-related info
    computationInfo.postMeshAdaptation();

    // Reset and copy to GPU the relevant data, if needed.
    computationInfo.cuda_resize();

    // Reset and copy to GPU the variables and Foo-data at cells
    cellFoo.cuda_resize(mesh.getCellCount());
    cellFoo.cuda_updateDevice();
#endif


//  adaptation::mapFields(parentIDs, currentIDs, parentCellFoo, cellFoo);

    //TODO: When OpenACC is on again, remove the following 4 lines updating
    //the host
//  cellFoo.cuda_updateHost();

    log_memory_status();
}


// This test takes a ScalarPiercedStorage, resizes it, increments on GPU its
// elements and copies it back to CPU to validate the sum of its elements.
// Tip: Run it with 2048 cells per direction, to cause its failure.
void test3(int argc, char *argv[])
{
    // Initialize process information
    int nProcessors;
    int rank;
#if ENABLE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nProcessors = 1;
    rank        = 0;
#endif

    // Initialize logger
    log::manager().initialize(log::COMBINED, "minimmerflow", true, ".", nProcessors, rank);
#if ENABLE_DEBUG==1
    log::cout().setVisibility(log::GLOBAL);
#endif

    // Log file header
    log::cout() << "o=====================================================o" << std::endl;
    log::cout() << "|                                                     |" << std::endl;
    log::cout() << "|                    minimmerflow  Solver                   |" << std::endl;
    log::cout() << "|                                                     |" << std::endl;
    log::cout() << "o=====================================================o" << std::endl;

    // Initialize configuration file
    config::reset("minimmerflow", 1);
    config::read("settings.xml");

    // Problem info
    const problem::ProblemType problemType = problem::getProblemType();

    int dimensions;
    double length;
    std::array<double, 3> origin;
    problem::getDomainData(problemType, dimensions, &origin, &length);

    const double tMin = problem::getStartTime(problemType, dimensions);
    const double tMax = problem::getEndTime(problemType, dimensions);

    log::cout() << std::endl;
    log::cout() << "Domain info: "  << std::endl;
    log::cout() << "  Origin .... " << origin << std::endl;
    log::cout() << "  Length .... " << length << std::endl;

    log::cout() << std::endl;
    log::cout() << "Time info: "  << std::endl;
    log::cout() << "  Initial .... " << tMin << std::endl;
    log::cout() << "  Final .... " << tMax << std::endl;

    // Discretization parameters
    const int order = config::root["discretization"]["space"].get<int>("order");

    const double cfl = config::root["discretization"]["time"].get<double>("CFL");

    long nCellsPerDirection;
    if (argc > 1) {
        nCellsPerDirection = atoi(argv[1]);
    } else {
        nCellsPerDirection = config::root["discretization"]["space"].get<long>("nCells");
    }

    log::cout() << std::endl;
    log::cout() << "Space distretization info..."  << std::endl;
    log::cout() << "  Order .... " << order << std::endl;
    log::cout() << "  Cells per direction .... " << nCellsPerDirection << std::endl;

    log::cout() << std::endl;
    log::cout() << "Time distretization info: "  << std::endl;
    log::cout() << "  CFL .... " << cfl << std::endl;

    // Output parametes
    int nSaves = std::numeric_limits<int>::max();
    if (argc > 2) {
        nSaves = atoi(argv[2]);
    }

    // Create the mesh
    log::cout() << std::endl;
    log::cout() << "Mesh initialization..."  << std::endl;

    unsigned int initialRefs = 0;
    const unsigned int maxInitialCellsProc = 1024;
    if (nProcessors>1) {
        while (pow(nCellsPerDirection , dimensions) > maxInitialCellsProc * nProcessors) {
            if ( (nCellsPerDirection%2) != 0) {
                break;
            }

            log::cout() << "Would create " << pow(nCellsPerDirection , dimensions) << " octants/processor. Reducing " << std::endl;
            nCellsPerDirection /= 2;
            initialRefs++;
        }
    }

    log::cout() << "*** Calling VolOctree constructor with " << nCellsPerDirection
                << " cells per direction, which will be uniformly refined " <<  initialRefs
                << " times." << std::endl;

#if ENABLE_MPI
    VolOctree mesh(dimensions, origin, length, length / nCellsPerDirection, MPI_COMM_WORLD);
#else
    VolOctree mesh(dimensions, origin, length, length / nCellsPerDirection);
#endif

    mesh.initializeAdjacencies();
    mesh.initializeInterfaces();

    mesh.update();

#if ENABLE_MPI
    if (nProcessors > 1) {
        mesh.partition(false, true);
    }
#endif

    for (unsigned int k=0; k<initialRefs; ++k){
        for (VolOctree::CellConstIterator cellItr = mesh.cellConstBegin(); cellItr != mesh.cellConstEnd(); ++cellItr) {
            mesh.markCellForRefinement(cellItr.getId());
        }

        log::cout() << "*** Mesh marked for refinement. Call update\n";
        mesh.update();
        log::cout() << "+++ mesh.update() DONE.\n";
        nCellsPerDirection *= 2;
    }

    {
        std::stringstream basename;
        basename << "background_" << nCellsPerDirection;
        mesh.getVTK().setName(basename.str().c_str());
    }

#if ENABLE_MPI
    if (nProcessors > 1) {
        log::cout() << "*** Call partition\n";
        mesh.partition(false, true);
        log::cout() << "+++ mesh.partition() DONE.\n";
    }
#endif
    mesh.write();

    log_memory_status();

    // Initialize body info
    body::initialize();

    // Initialize computation data
    log::cout() << std::endl;
    log::cout() << "Computation data initialization..."  << std::endl;

    ComputationInfo computationInfo(&mesh);
#if ENABLE_CUDA
    computationInfo.cuda_initialize();
#endif

    ScalarPiercedStorage<std::size_t> cellFoo;
//  cellFoo.rawReserve(4 * (&mesh.getCellCount()));
    cellFoo.setDynamicKernel(&mesh.getCells(), PiercedVector<Cell>::SYNC_MODE_JOURNALED);
    cellFoo.cuda_allocateDevice();

    std::cout << "Bef cellFoo.data()" << cellFoo.data() << std::endl;
    cellFoo.cuda_fillDevice(1);

    cellFoo.cuda_updateHost();

    std::size_t sum = 0;
    for (int iter = 0; iter < cellFoo.cuda_deviceDataSize(); iter++) {
        sum += cellFoo[iter];
    }
    std::cout << "validated 1st sum = " << cellFoo.cuda_deviceDataSize()
              << " and sum = " << sum << std::endl;



    log_memory_status();

    // Initialize storage
    log::cout() << std::endl;
    log::cout() << "Storage initialization..."  << std::endl;

    log_memory_status();

    // Initialize reconstruction
    log::cout() << std::endl;
    log::cout() << "Reconstruction initialization..."  << std::endl;

    log_memory_status();

    adaptMeshAndFields(computationInfo, mesh, cellFoo, problemType);
    std::cout << "Aft cellFoo.data()" << cellFoo.data() << std::endl;
    std::cout << "main 0" << std::endl;

    std::cout << "cellFoo.data()" << cellFoo.data() << std::endl;
    cellFoo.cuda_fillDevice(1);
    test::plotPiercedStorage(cellFoo, cellFoo.cuda_deviceDataSize());
    std::cout << "cellFoo.data()" << cellFoo.data() << std::endl;

    cellFoo.cuda_updateHost();

    sum = 0;
    for (int iter = 0; iter < cellFoo.cuda_deviceDataSize(); iter++) {
        sum += cellFoo[iter];
    }
    std::cout << "validated 2nd sum = " << 2 * cellFoo.cuda_deviceDataSize()
              << " and sum = " << sum << std::endl;

    if (sum == 2 * cellFoo.cuda_deviceDataSize()) {
        std::cout << "\nTEST #3: SUCCESSFULL" << std::endl;
    } else {
        std::cout << "\nTEST #3: UNSUCCESSFULL" << std::endl;
    }

    log_memory_status();

    // Clean-up
#if ENABLE_CUDA
    cellFoo.cuda_freeDevice();
    computationInfo.cuda_finalize();
#endif

}

int main(int argc, char *argv[])
{

    bool basicTest = false;
    if (basicTest) {
        test0RT();
    } else {

        // Get the primary context and bind to it
        CUcontext ctx;
        CUdevice dev;

        CHECK_DRV(cuInit(0));
        CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, 0));
        CHECK_DRV(cuCtxSetCurrent(ctx));
        CHECK_DRV(cuCtxGetDevice(&dev));

    //  CUcontext context{0};
    //  CUDA_DRIVER_ERROR_CHECK(cuCtxGetCurrent(&context));
    //  CUdevice device;
    //  CUDA_DRIVER_ERROR_CHECK(cuCtxGetDevice(&device));
    //  CUcontext primary_context;
    //  CUDA_DRIVER_ERROR_CHECK(cuDevicePrimaryCtxRetain(&primary_context, device));
    //  unsigned int flags;
    //  int active;
    //  CUresult primary_state_check_result = cuDevicePrimaryCtxGetState(device, &flags, &active);


        //
        // Initialization
        //

#if ENABLE_MPI
        // MPI Initialization
        MPI_Init(&argc, &argv);
#endif

        bool runTest0 = false;
        bool runTest1A = true;
        bool runTest1B = false;
        bool runTest2 = false;
        bool runTest3 = false;

        // test0
        if (runTest0) {
            try{
                std::cout << "EXECUTING TEST #0DR" << std::endl;
                test0DR();
                std::cout << "\n" << std::endl;
            }
            catch(std::exception & e){
                std::cout << "TEST #0DR exited with an error of type : " << e.what() << std::endl;
                return 1;
            }
        }

        // test1A
        if (runTest1A) {
            try{
                std::cout << "EXECUTING TEST #1A" << std::endl;
                test1A();
                std::cout << "\n" << std::endl;
            }
            catch(std::exception & e){
                std::cout << "TEST #1A exited with an error of type : " << e.what() << std::endl;
                return 1;
            }
        }

        // test1B
        if (runTest1B) {
            try{
                std::cout << "EXECUTING TEST #1B" << std::endl;
                test1B();
                std::cout << "\n" << std::endl;
            }
            catch(std::exception & e){
                std::cout << "TEST #1B exited with an error of type : " << e.what() << std::endl;
                return 1;
            }
        }

        // test2
        if (runTest2) {
            try{
                std::cout << "EXECUTING TEST #2" << std::endl;
                test2();
                std::cout << "\n" << std::endl;
            }
            catch(std::exception & e){
                std::cout << "TEST #2 exited with an error of type : " << e.what() << std::endl;
                return 1;
            }
        }


        // test3
        if (runTest3) {
            try{
                std::cout << "EXECUTING TEST #3" << std::endl;
                test3(argc, argv);
                std::cout << "\n" << std::endl;
            }
            catch(std::exception & e){
                std::cout << "TEST #3 exited with an error of type : " << e.what() << std::endl;
                return 1;
            }
        }

        //
        // Finalization
        //

#if ENABLE_MPI
        // MPI finalization
        MPI_Finalize();
#endif
    }
}
