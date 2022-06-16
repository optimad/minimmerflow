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
#include "communications.hpp"

#include <bitpit_IO.hpp>
#include <bitpit_voloctree.hpp>

#if ENABLE_MPI
#include <mpi.h>
#endif
#include <vector>
#include <time.h>

#include <nvtx3/nvToolsExt.h>
#include "cuda_runtime_api.h"

using namespace bitpit;

void computation(int argc, char *argv[])
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

#if ENABLE_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);               // How many GPUs?
    int device_id = rank % deviceCount;
    cudaSetDevice(device_id);                       // Map MPI-process to a GPU
    std::cout << "Rank " << rank << " uses GPU device #" << device_id << std::endl;
#endif

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

    const ScalarPiercedStorage<int> &cellSolveMethods = computationInfo.getCellSolveMethods();

    const ScalarStorage<std::size_t> &solvedCellRawIds = computationInfo.getSolvedCellRawIds();
    const std::size_t nSolvedCells = solvedCellRawIds.size();

    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceRawIds = computationInfo.getSolvedBoundaryInterfaceRawIds();
    const std::size_t nSolvedBoundaryInterfaces = solvedBoundaryInterfaceRawIds.size();

    log_memory_status();

    // Initialize storage
    log::cout() << std::endl;
    log::cout() << "Storage initialization..."  << std::endl;

    ScalarPiercedStorageCollection<double> cellPrimitives(N_FIELDS, &(mesh.getCells()));
    ScalarPiercedStorageCollection<double> cellConservatives(N_FIELDS, &(mesh.getCells()));
    ScalarPiercedStorageCollection<double> cellConservativesWork(N_FIELDS, &(mesh.getCells()));
    ScalarPiercedStorageCollection<double> cellRHS(N_FIELDS, &(mesh.getCells()));

#if ENABLE_CUDA
    cellRHS.cuda_allocateDevice();
    cellConservatives.cuda_allocateDevice();
    cellConservativesWork.cuda_allocateDevice();
    cellPrimitives.cuda_allocateDevice();
#endif

    // Set host storage pointers for OpenACC
    double** cellPrimitivesHostStorageCollection = cellPrimitives.collectionData();
    double** cellConservativesHostStorageCollection = cellConservatives.collectionData();
    double** cellConservativesWorkHostStorageCollection = cellConservativesWork.collectionData();
    double** cellRHSHostStorageCollection = cellRHS.collectionData();
    double *cellVolumeHostStorage = computationInfo.getCellVolumes().data();
    const std::size_t *solvedCellRawIdsHostStorage = solvedCellRawIds.data();

    log_memory_status();

    // Initialize reconstruction
    log::cout() << std::endl;
    log::cout() << "Reconstruction initialization..."  << std::endl;

    reconstruction::initialize();
    log_memory_status();

#if ENABLE_CUDA
    // Initialize Euler solver
    euler::cuda_initialize();
#endif

    // Boundary conditions
    log::cout() << std::endl;
    log::cout() << "Boundary conditions initialization..."  << std::endl;

    ScalarStorage<int> solvedBoundaryInterfaceBCs(nSolvedBoundaryInterfaces);

    for (std::size_t i = 0; i < nSolvedBoundaryInterfaces; ++i) {
        const std::size_t interfaceRawId = solvedBoundaryInterfaceRawIds[i];
        const Interface &interface = mesh.getInterfaces().rawAt(interfaceRawId);

        bool isIntrBorder = interface.isBorder();
        if (isIntrBorder) {
            long interfaceId = interface.getId();
            solvedBoundaryInterfaceBCs[i] = problem::getBorderBCType(problemType, interfaceId, computationInfo);
        } else {
            solvedBoundaryInterfaceBCs[i] = BC_WALL;
        }
    }


#if ENABLE_CUDA
    solvedBoundaryInterfaceBCs.cuda_allocateDevice();
    solvedBoundaryInterfaceBCs.cuda_updateDevice();
#endif

    log_memory_status();

    // Initialize output
    log::cout() << std::endl;
    log::cout() << "Output initialization..."  << std::endl;

    SolverWriter solverStreamer(&mesh, &cellSolveMethods, &cellPrimitives, &cellConservatives, &cellRHS);

    VTKUnstructuredGrid &vtk = mesh.getVTK();
    vtk.setCounter(0);

    vtk.addData<int>("solveMethod"   , VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("velocity"   , VTKFieldType::VECTOR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("pressure"   , VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("temperature", VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("density"    , VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("residualC"  , VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("residualMX", VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("residualMY", VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("residualMZ", VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
    vtk.addData<double>("residualE", VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);

    log_memory_status();


#if ENABLE_CUDA==1
    std::unordered_map<int, ScalarStorage<std::size_t>> sourcesListsMap;
    //std::unordered_map<int, ScalarStorageCollection<double>> sourcesValuesMap;
    std::vector<std::unordered_map<int, ScalarStorage<double>>> conservativeWorkSourceValuesMap(N_FIELDS);
    std::vector<std::unordered_map<int, ScalarStorage<double>>> conservativeSourceValuesMap(N_FIELDS);
    std::vector<std::unordered_map<int, ScalarStorage<double>>> primitiveSourceValuesMap(N_FIELDS);
#endif
#if ENABLE_MPI
    // Creating ghost communications for exchanging solved data
    log::cout() << std::endl;
    log::cout() << " * Inizializing ghost communications for exchanging ghost data" << std::endl;

    std::unique_ptr<GhostCommunicator> primitiveCommunicator;
    std::unique_ptr<GhostCommunicator> conservativeCommunicator;
    std::unique_ptr<GhostCommunicator> init_conservativeCommunicator;
    std::unique_ptr<GhostCommunicator> conservativeWorkCommunicator;

#if ENABLE_CUDA==1
    std::vector<std::unique_ptr<CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>>> primitiveGhostWriteStreamers(N_FIELDS);
    std::vector<std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>> primitiveGhostReadStreamers(N_FIELDS);
#else
    std::vector<std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>> primitiveGhostStreamers(N_FIELDS);
#endif
#if ENABLE_CUDA==1
    std::vector<std::unique_ptr<CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>>> conservativeGhostWriteStreamers(N_FIELDS);
    std::vector<std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>> conservativeGhostReadStreamers(N_FIELDS);
    std::vector<std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>> init_conservativeGhostStreamers(N_FIELDS);
#else
    std::vector<std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>> conservativeGhostStreamers(N_FIELDS);
#endif
#if ENABLE_CUDA==1
    std::vector<std::unique_ptr<CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>>> conservativeWorkGhostWriteStreamers(N_FIELDS);
    std::vector<std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>> conservativeWorkGhostReadStreamers(N_FIELDS);
#else
    std::vector<std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>> conservativeWorkGhostStreamers(N_FIELDS);
#endif
    if (mesh.isPartitioned()) {
        // Primitive fields
        primitiveCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh, "primitive"));
        primitiveCommunicator->resetExchangeLists();
        primitiveCommunicator->setRecvsContinuous(true);

        size_t writeBufferSize = 0;
        for(auto & rankList : primitiveCommunicator->getSendList()) {
            if (writeBufferSize < rankList.second.size()) {
                writeBufferSize = rankList.second.size();
            }
        }
        std::cout << "Max buffer size = " << writeBufferSize << std::endl;


        for (int k = 0; k < N_FIELDS; ++k) {
#if ENABLE_CUDA==1
            primitiveGhostReadStreamers[k] = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&(cellPrimitives[k])));
            primitiveGhostWriteStreamers[k] = std::unique_ptr<CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>>(
                new CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>(&(primitiveSourceValuesMap[k]), 0, 0, sizeof(ScalarStorage<double>::value_type), writeBufferSize));
            primitiveCommunicator->addData(primitiveGhostWriteStreamers[k].get(), primitiveGhostReadStreamers[k].get());
#else
            primitiveGhostStreamers[k] = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&(cellPrimitives[k])));
            primitiveCommunicator->addData(primitiveGhostStreamers[k].get());
#endif
        }

        // Conservative fields
        conservativeCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh, "conservative"));
        conservativeCommunicator->resetExchangeLists();
        conservativeCommunicator->setRecvsContinuous(true);
        init_conservativeCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh, "init_conservative"));
        init_conservativeCommunicator->resetExchangeLists();
        init_conservativeCommunicator->setRecvsContinuous(true);

        for (int k = 0; k < N_FIELDS; ++k) {
#if ENABLE_CUDA==1
            conservativeGhostReadStreamers[k] = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&(cellConservatives[k])));
            conservativeGhostWriteStreamers[k] = std::unique_ptr<CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>>(
                new CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>(&(conservativeSourceValuesMap[k]), 0, 0, sizeof(ScalarStorage<double>::value_type), writeBufferSize));
            conservativeCommunicator->addData(conservativeGhostWriteStreamers[k].get(), conservativeGhostReadStreamers[k].get());
            init_conservativeGhostStreamers[k] = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&(cellConservatives[k])));
            init_conservativeCommunicator->addData(init_conservativeGhostStreamers[k].get());
#else
            conservativeGhostStreamers[k] = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&(cellConservatives[k])));
            conservativeCommunicator->addData(conservativeGhostStreamers[k].get());
#endif
        }

        // Conservative tields tmp
        conservativeWorkCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh, "conservative work"));
        conservativeWorkCommunicator->resetExchangeLists();
        conservativeWorkCommunicator->setRecvsContinuous(true);

        for (int k = 0; k < N_FIELDS; ++k) {

#if ENABLE_CUDA==1
            conservativeWorkGhostReadStreamers[k] = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&(cellConservativesWork[k])));
            conservativeWorkGhostWriteStreamers[k] = std::unique_ptr<CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>>(
                new CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>(&(conservativeWorkSourceValuesMap[k]), 0, 0, sizeof(ScalarStorage<double>::value_type), writeBufferSize));
            conservativeWorkCommunicator->addData(conservativeWorkGhostWriteStreamers[k].get(), conservativeWorkGhostReadStreamers[k].get());
#else
            conservativeWorkGhostStreamers[k] = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&(cellConservativesWork[k])));
            conservativeWorkCommunicator->addData(conservativeWorkGhostStreamers[k].get());
#endif
        }
    }

#if ENABLE_CUDA
    const std::vector<int> &sendRanks = conservativeWorkCommunicator->getSendRanks();
    MPI_Barrier(MPI_COMM_WORLD);
    std::size_t nSendRanks = sendRanks.size();
    for (std::size_t n = 0; n < nSendRanks; ++n) {
        int r = sendRanks[n];
        const ListCommunicator::RankExchangeList & rankList = conservativeWorkCommunicator->getSendList(r);
        // Store sources raw ids list on device
        sourcesListsMap[r] = ScalarStorage<std::size_t>(rankList.size());
        ScalarStorage<std::size_t> & sourceList = sourcesListsMap[r];
        const auto & rankSources = mesh.getGhostCellExchangeSources(r);
        for (std::size_t i = 0; i < rankList.size(); ++i) {
            long cellId = rankSources[rankList[i]];
            sourceList[i] = mesh.getCellIterator(cellId).getRawIndex();
        }
        sourceList.cuda_allocateDevice();
        sourceList.cuda_updateDevice();
        // Allocate source value storages on device
        for (std::size_t k = 0; k < N_FIELDS; ++k) {
            ScalarStorage<double> & primitiveSourceValues = primitiveSourceValuesMap[k][r];
            ScalarStorage<double> & conservativeSourceValues = conservativeSourceValuesMap[k][r];
            ScalarStorage<double> & conservativeWorkSourceValues = conservativeWorkSourceValuesMap[k][r];
            primitiveSourceValues = ScalarStorage<double>(rankList.size());
            conservativeSourceValues = ScalarStorage<double>(rankList.size());
            conservativeWorkSourceValues = ScalarStorage<double>(rankList.size());
            primitiveSourceValues.cuda_allocateDevice();
            conservativeSourceValues.cuda_allocateDevice();
            conservativeWorkSourceValues.cuda_allocateDevice();
        }
    }
#endif

#endif

    // Initial conditions - Conservatives initializion on CPU, Primitives initialization on GPU - TODO initial consrvatives on GPU
    log::cout() << std::endl;
    log::cout() << "Initial conditions evaluation..."  << std::endl;

    for (std::size_t i = 0; i < nSolvedCells; ++i) {
        const std::size_t cellRawId = solvedCellRawIds[i];
        const long cellId = mesh.getCells().rawFind(cellRawId).getId();

        std::array<double, N_FIELDS> conservatives;
        problem::evalCellInitalConservatives(problemType, cellId, computationInfo, conservatives.data());
        for (int k = 0; k < N_FIELDS; ++k) {
            cellConservatives[k].rawAt(cellRawId) = conservatives[k];
        }
    }

#if ENABLE_CUDA
    // It can be avoided by computing conservatives on GPU
    cellConservatives.cuda_updateDevice();
#endif

#pragma acc parallel loop present(cellPrimitivesHostStorageCollection, cellConservativesHostStorageCollection, solvedCellRawIdsHostStorage)
    for (long i = 0; i < nSolvedCells; ++i) {
      const std::size_t cellRawId = solvedCellRawIdsHostStorage[i];
      std::array<double, N_FIELDS> conservatives;
      std::array<double, N_FIELDS> primitives;
#pragma acc loop seq
      for (int k = 0; k < N_FIELDS; ++k) {
          conservatives[k] = cellConservativesHostStorageCollection[k][cellRawId];
      }
      ::utils::conservative2primitive(conservatives.data(), primitives.data());
#pragma acc loop seq
      for (int k = 0; k < N_FIELDS; ++k) {
          cellPrimitivesHostStorageCollection[k][cellRawId] = primitives[k];
      }
    }

#if ENABLE_CUDA && ENABLE_MPI
    // This update is not for communication only and it must happen
    cellPrimitives.cuda_updateHost();
#endif

#if ENABLE_CUDA && ENABLE_MPI
    // Communication streamer needs GPU values buffer to be correctly initialized
    for (int i = 0; i < nSendRanks; ++i) {
        int r = sendRanks[i];
        std::size_t listSize = sourcesListsMap[r].size();
        std::size_t *rankList = sourcesListsMap[r].data();
        for (int k = 0; k < N_FIELDS; ++k) {
            double *primitiveRankValues = primitiveSourceValuesMap[k][r].data();
            double *conservativeRankValues = conservativeSourceValuesMap[k][r].data();
#pragma acc parallel loop present(cellPrimitivesHostStorageCollection, cellConservativesHostStorageCollection, rankList, primitiveRankValues, conservativeRankValues)
//#pragma acc parallel loop present(cellConservativesHostStorageCollection, rankList, conservativeRankValues)
            for (std::size_t i = 0; i < listSize; ++i) {
                const std::size_t cellRawId = rankList[i];
                double *primitiveRankValue = &primitiveRankValues[i];
                double *conservativeRankValue = &conservativeRankValues[i];
                const double* conservativeTmp = &cellConservativesHostStorageCollection[k][cellRawId];
                const double* primitiveTmp = &cellPrimitivesHostStorageCollection[k][cellRawId];
                *primitiveRankValue = *primitiveTmp;
                *conservativeRankValue = *conservativeTmp;
            }
        }
    }
#endif

#if ENABLE_MPI
    log::cout() << "Initial communication..."  << std::endl;
    for (int k = 0; k < N_FIELDS; ++k) {
        primitiveGhostWriteStreamers[k].get()->initializeCUDAObjects();
        conservativeGhostWriteStreamers[k].get()->initializeCUDAObjects();
        conservativeWorkGhostWriteStreamers[k].get()->initializeCUDAObjects();
    }



    if (mesh.isPartitioned()) {
        log::cout() << "Conservative cuda objects initialization" << std::endl;
        conservativeCommunicator->initializeCudaObjects();
        log::cout() << "Conservative WORK cuda objects initialization" << std::endl;
        conservativeWorkCommunicator->initializeCudaObjects();
        log::cout() << "Primitive cuda objects initialization" << std::endl;
        primitiveCommunicator->initializeCudaObjects();

        conservativeCommunicator->startAllExchanges();
        primitiveCommunicator->startAllExchanges();
        conservativeCommunicator->completeAllExchanges();
        primitiveCommunicator->completeAllExchanges();
    }
#endif

    mesh.write();

    log_memory_status();

    // Find smallest cell
    double minCellSize = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < nSolvedCells; ++i) {
        const std::size_t cellRawId = solvedCellRawIds[i];

        minCellSize = std::min(computationInfo.rawGetCellSize(cellRawId), minCellSize);
    }

#if ENABLE_MPI
    if (mesh.isPartitioned()) {
        MPI_Allreduce(MPI_IN_PLACE, &minCellSize, 1, MPI_DOUBLE, MPI_MIN, mesh.getCommunicator());
    }
#endif

    log_memory_status();

    // Start computation
    log::cout() << std::endl;
    log::cout() << "Starting computation..."  << std::endl;
    log::cout() << std::endl;

    clock_t diskTime = 0;
    clock_t computeStart = clock();

    int step = 0;
    double t = tMin;
    double nextSave = tMin;
    while (t < tMax && step < 3) {
        log::cout() << std::endl;
        log::cout() << "Step n. " << step << std::endl;

        double maxEig;

        std::string rangeName = "TimeStep" + std::to_string(step);
        nvtxRangePushA(rangeName.c_str());

        //
        // FIRST RK STAGE
        //
        nvtxRangePushA("RK1");

        // Compute the residuals
#if ENABLE_CUDA && ENABLE_MPI
        if (nProcessors > 1) {
            nvtxRangePushA("RK1_MPI_DU");
            long firstGhostId = mesh.getFirstGhostCell().getId();
            long firstGhostRawId = mesh.getCellIterator(firstGhostId).getRawIndex();
            for (int i = 0; i < N_FIELDS; ++i) {
                cellConservatives[i].cuda_updateDevice(cellConservatives[i].rawFind(firstGhostRawId), cellConservatives[i].end());
            }
//            if(step == 2) {
//                std::stringstream ss;
//                ss << "ghosts_" << rank << ".txt";
//                std::ofstream out(ss.str().c_str());
//                if (rank == 0) {
//                    for (int i = 0; i < N_FIELDS; ++i) {
//                        std::cout << "RANK = " << rank << " FIELD = " << i << std::endl;
//                        out << "RANK = " << rank << " FIELD = " << i << std::endl;
//                        auto begin = cellConservatives[i].rawFind(firstGhostRawId);
//                        auto end = cellConservatives[i].end();
//                        for (auto it = begin; it != end; ++it ) {
//                            std::cout << *it  << "(" << rank<< ")" << " ";
//                            out << *it << " ";
//                        }
//                        std::cout << std::flush << std::endl;
//                        out << std::flush << std::endl;
//                    }
//                }
//                MPI_Barrier(MPI_COMM_WORLD);
//                if (rank == 1) {
//                    for (int i = 0; i < N_FIELDS; ++i) {
//                        std::cout << "RANK = " << rank << " FIELD = " << i << std::endl;
//                        out << "RANK = " << rank << " FIELD = " << i << std::endl;
//                        auto begin = cellConservatives[i].rawFind(firstGhostRawId);
//                        auto end = cellConservatives[i].end();
//                        for (auto it = begin; it != end; ++it ) {
//                            std::cout << *it  << "(" << rank<< ")" << " ";
//                            out << *it << " ";
//                        }
//                        std::cout << std::flush << std::endl;
//                        out << std::flush << std::endl;
//                    }
//                }
//                out.close();
//
//            }
            nvtxRangePop();
        }
#endif

        nvtxRangePushA("computePolynomials");
        reconstruction::computePolynomials(problemType, computationInfo, cellConservatives, solvedBoundaryInterfaceBCs);
        nvtxRangePop();

        nvtxRangePushA("computeRHS");
        euler::computeRHS(problemType, computationInfo, order, solvedBoundaryInterfaceBCs, cellConservatives, &cellRHS, &maxEig);
        nvtxRangePop();

#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            MPI_Allreduce(MPI_IN_PLACE, &maxEig, 1, MPI_DOUBLE, MPI_MAX, mesh.getCommunicator());
        }
#endif

        // Choose dt
        double dt = 0.9 * cfl * minCellSize / maxEig;
        log::cout() << "Ideal dt = " << dt << " (maxEig = " << maxEig << ")" <<std::endl;
        if (t + dt > tMax) {
            dt = tMax - t;
        }
        log::cout() << "Using dt= " << dt <<std::endl;
        log::cout() << "Current time= " << (t + dt) << std::endl;

        nvtxRangePop(); // pop RK1 range

        //
        // SECOND RK STAGE
        //
        nvtxRangePushA("RK2");

        nvtxRangePushA("OpenACC_RK2_updateSolution");
#pragma acc parallel loop collapse(2) present(cellVolumeHostStorage, solvedCellRawIdsHostStorage, cellConservativesHostStorageCollection, cellConservativesWorkHostStorageCollection, cellRHSHostStorageCollection)
        for (std::size_t i = 0; i < nSolvedCells; ++i) {
            for (int k = 0; k < N_FIELDS; ++k) {
                const std::size_t cellRawId = solvedCellRawIdsHostStorage[i];
                const double cellVolume = cellVolumeHostStorage[cellRawId];
                const double RHS = cellRHSHostStorageCollection[k][cellRawId];
                const double conservative = cellConservativesHostStorageCollection[k][cellRawId];
                double *conservativeTmp = &cellConservativesWorkHostStorageCollection[k][cellRawId];
                *conservativeTmp = conservative + dt * RHS / cellVolume;
            }
        }
        nvtxRangePop();

#if ENABLE_CUDA && ENABLE_MPI
        if (nProcessors > 1) {
            nvtxRangePushA("RK2_MPI_HU");

            for (int i = 0; i < nSendRanks; ++i) {
                int r = sendRanks[i];
                std::size_t listSize = sourcesListsMap[r].size();
                std::size_t *rankList = sourcesListsMap[r].data();
                for (int k = 0; k < N_FIELDS; ++k) {
                    double *rankValues = conservativeWorkSourceValuesMap[k][r].data();
#pragma acc parallel loop present(cellConservativesWorkHostStorageCollection, rankList, rankValues)
                    for (std::size_t i = 0; i < listSize; ++i) {
                        const std::size_t cellRawId = rankList[i];
                        double *rankValue = &rankValues[i];
                        const double* conservativeTmp = &cellConservativesWorkHostStorageCollection[k][cellRawId];
                        *rankValue = *conservativeTmp;
                    }
                }
            }

            nvtxRangePop();
        }
#endif

#if ENABLE_MPI
        nvtxRangePushA("RK2_COMM");
        if (mesh.isPartitioned()) {
            conservativeWorkCommunicator->startAllExchanges();
            conservativeWorkCommunicator->completeAllExchanges();
        }
        nvtxRangePop();
#endif

#if ENABLE_CUDA && ENABLE_MPI
        if (nProcessors > 1) {
            nvtxRangePushA("RK2_MPI_DU");
            long firstGhostId = mesh.getFirstGhostCell().getId();
            long firstGhostRawId = mesh.getCellIterator(firstGhostId).getRawIndex();
            for (int i = 0; i < N_FIELDS; ++i) {
                cellConservativesWork[i].cuda_updateDevice(cellConservativesWork[i].rawFind(firstGhostRawId), cellConservativesWork[i].end());
            }
//            std::stringstream ss;
//            ss << "ghosts_" << rank << ".txt";
//            std::ofstream out(ss.str().c_str());
//            if (rank == 0) {
//                for (int i = 0; i < N_FIELDS; ++i) {
//                    std::cout << "RANK = " << rank << " FIELD = " << i << std::endl;
//                    out << "RANK = " << rank << " FIELD = " << i << std::endl;
//                    auto begin = cellConservativesWork[i].rawFind(firstGhostRawId);
//                    auto end = cellConservativesWork[i].end();
//                    for (auto it = begin; it != end; ++it ) {
//                        std::cout << *it  << "(" << rank<< ")" << " ";
//                        out << *it << " ";
//                    }
//                    std::cout << std::flush << std::endl;
//                    out << std::flush << std::endl;
//                }
//            }
//            MPI_Barrier(MPI_COMM_WORLD);
//            if (rank == 1) {
//                for (int i = 0; i < N_FIELDS; ++i) {
//                    std::cout << "RANK = " << rank << " FIELD = " << i << std::endl;
//                    out << "RANK = " << rank << " FIELD = " << i << std::endl;
//                    auto begin = cellConservativesWork[i].rawFind(firstGhostRawId);
//                    auto end = cellConservativesWork[i].end();
//                    for (auto it = begin; it != end; ++it ) {
//                        std::cout << *it  << "(" << rank<< ")" << " ";
//                        out << *it << " ";
//                    }
//                    std::cout << std::flush << std::endl;
//                    out << std::flush << std::endl;
//                }
//            }
//            out.close();
            nvtxRangePop();
        }
#endif

        nvtxRangePushA("ComputePolynomials");
        reconstruction::computePolynomials(problemType, computationInfo, cellConservativesWork, solvedBoundaryInterfaceBCs);
        nvtxRangePop();

        nvtxRangePushA("computeRHS");
        euler::computeRHS(problemType, computationInfo, order, solvedBoundaryInterfaceBCs, cellConservativesWork, &cellRHS, &maxEig);
        nvtxRangePop();

#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            MPI_Allreduce(MPI_IN_PLACE, &maxEig, 1, MPI_DOUBLE, MPI_MAX, mesh.getCommunicator());
        }
#endif

        log::cout() << "(maxEig after second stage = " << maxEig << ")" << std::endl;
        nvtxRangePop(); // pop RK2 range

        //
        // THIRD RK STAGE
        //
        nvtxRangePushA("RK3");

        nvtxRangePushA("OpenACC_RK3_updateSolution");
#pragma acc parallel loop collapse(2) present(cellVolumeHostStorage, solvedCellRawIdsHostStorage, cellConservativesHostStorageCollection, cellConservativesWorkHostStorageCollection, cellRHSHostStorageCollection)
        for (std::size_t i = 0; i < nSolvedCells; ++i) {
            for (int k = 0; k < N_FIELDS; ++k) {
                const std::size_t cellRawId = solvedCellRawIdsHostStorage[i];
                const double cellVolume = cellVolumeHostStorage[cellRawId];
                const double RHS = cellRHSHostStorageCollection[k][cellRawId];
                const double conservative = cellConservativesHostStorageCollection[k][cellRawId];
                double *conservativeWork = &cellConservativesWorkHostStorageCollection[k][cellRawId];
                *conservativeWork = 0.75 * conservative + 0.25 * ((*conservativeWork) + dt * RHS / cellVolume);
            }
        }
        nvtxRangePop();

#if ENABLE_CUDA && ENABLE_MPI
        if (nProcessors > 1) {
            nvtxRangePushA("RK3_MPI_HU");
            for (int i = 0; i < nSendRanks; ++i) {
                int r = sendRanks[i];
                std::size_t listSize = sourcesListsMap[r].size();
                std::size_t *rankList = sourcesListsMap[r].data();
                for (int k = 0; k < N_FIELDS; ++k) {
                    double *rankValues = conservativeWorkSourceValuesMap[k][r].data();
#pragma acc parallel loop present(cellConservativesWorkHostStorageCollection, rankList, rankValues)
                    for (std::size_t i = 0; i < listSize; ++i) {
                        const std::size_t cellRawId = rankList[i];
                        double *rankValue = &rankValues[i];
                        const double* conservativeTmp = &cellConservativesWorkHostStorageCollection[k][cellRawId];
                        *rankValue = *conservativeTmp;
                    }
                }
            }
            nvtxRangePop();
        }
#endif

#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            conservativeWorkCommunicator->startAllExchanges();
            conservativeWorkCommunicator->completeAllExchanges();
        }
#endif

#if ENABLE_CUDA
        if (nProcessors > 1) {
            nvtxRangePushA("RK3_MPI_DU");
            long firstGhostId = mesh.getFirstGhostCell().getId();
            long firstGhostRawId = mesh.getCellIterator(firstGhostId).getRawIndex();
            for (int i = 0; i < N_FIELDS; ++i) {
                cellConservativesWork[i].cuda_updateDevice(cellConservativesWork[i].rawFind(firstGhostRawId), cellConservativesWork[i].end());
            }
            nvtxRangePop();
        }
#endif

        nvtxRangePushA("ComputePolynomials");
        reconstruction::computePolynomials(problemType, computationInfo, cellConservativesWork, solvedBoundaryInterfaceBCs);
        nvtxRangePop();

        nvtxRangePushA("computeRHS");
        euler::computeRHS(problemType, computationInfo, order, solvedBoundaryInterfaceBCs, cellConservativesWork, &cellRHS, &maxEig);
        nvtxRangePop();

#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            MPI_Allreduce(MPI_IN_PLACE, &maxEig, 1, MPI_DOUBLE, MPI_MAX, mesh.getCommunicator());
        }
#endif

        log::cout() << "(maxEig after third stage = " << maxEig << ")" << std::endl;
        nvtxRangePop(); // pop RK3 range

        //
        // CLOSE RK STEP
        //
        nvtxRangePushA("RKFinal");
        nvtxRangePushA("OpenACC_RKFinal_updateSolution");
#pragma acc parallel loop collapse(2) present(cellVolumeHostStorage, solvedCellRawIdsHostStorage, cellConservativesHostStorageCollection, cellConservativesWorkHostStorageCollection, cellRHSHostStorageCollection)
        for (std::size_t i = 0; i < nSolvedCells; ++i) {
            for (int k = 0; k < N_FIELDS; ++k) {
                const std::size_t cellRawId = solvedCellRawIdsHostStorage[i];
                const double cellVolume = cellVolumeHostStorage[cellRawId];
                const double RHS = cellRHSHostStorageCollection[k][cellRawId];
                double *conservative = &cellConservativesHostStorageCollection[k][cellRawId];
                const double conservativeWork = cellConservativesWorkHostStorageCollection[k][cellRawId];
                *conservative = (1. / 3) * (*conservative) + (2. / 3) * (conservativeWork + dt * RHS / cellVolume);
            }
        }
        nvtxRangePop();

#if ENABLE_CUDA && ENABLE_MPI
        if (nProcessors > 1) {
            nvtxRangePushA("RKFinal_MPI_HU");
//            cellConservatives.cuda_updateHost();
            for (int i = 0; i < nSendRanks; ++i) {
                int r = sendRanks[i];
                std::size_t listSize = sourcesListsMap[r].size();
                std::size_t *rankList = sourcesListsMap[r].data();
                for (int k = 0; k < N_FIELDS; ++k) {
                    double *rankValues = conservativeSourceValuesMap[k][r].data();
#pragma acc parallel loop present(cellConservativesHostStorageCollection, rankList, rankValues)
                    for (std::size_t i = 0; i < listSize; ++i) {
                        const std::size_t cellRawId = rankList[i];
                        double *rankValue = &rankValues[i];
                        const double* conservativeTmp = &cellConservativesHostStorageCollection[k][cellRawId];
                        *rankValue = *conservativeTmp;
                    }
                }
            }
            nvtxRangePop();
        }
#endif

#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            conservativeCommunicator->startAllExchanges();
            conservativeCommunicator->completeAllExchanges();
//            init_conservativeCommunicator->startAllExchanges();
//            init_conservativeCommunicator->completeAllExchanges();
        }
#endif
        nvtxRangePop(); // pop RKFinal range

        // Update timestep information
        t +=dt;
        step++;

        // Write the solution
        // if (t > nextSave){
        //     clock_t diskStart = clock();
        //
        //     std::array<double, N_FIELDS> conservatives;
        //     std::array<double, N_FIELDS> primitives;
        //
        //     for (std::size_t i = 0; i < nSolvedCells; ++i) {
        //         const std::size_t cellRawId = solvedCellRawIds[i];
        //
        //         for (int k = 0; k < N_FIELDS; ++k) {
        //             conservatives[k] = cellConservatives[k].rawAt(cellRawId);
        //         }
        //
        //         ::utils::conservative2primitive(conservatives.data(), primitives.data());
        //
        //         for (int k = 0; k < N_FIELDS; ++k) {
        //             cellPrimitives[k].rawAt(cellRawId) = primitives[k];
        //         }
        //     }
        //     mesh.write();
        //
        //     diskTime += clock() - diskStart;
        //     nextSave += (tMax - tMin) / nSaves;
        // }
        nvtxRangePop(); // pop TimeStep range
    }
    clock_t computeEnd = clock();

    if (nProcessors > 1) {
        long firstGhostId = mesh.getFirstGhostCell().getId();
        long firstGhostRawId = mesh.getCellIterator(firstGhostId).getRawIndex();
        for (int i = 0; i < N_FIELDS; ++i) {
            cellConservatives[i].cuda_updateDevice(cellConservatives[i].rawFind(firstGhostRawId), cellConservatives[i].end());
        }
        cellConservatives.cuda_updateHost();
    }
    // Save final data
    {
        std::array<double, N_FIELDS> conservatives;
        std::array<double, N_FIELDS> primitives;

        std::stringstream filename;
        filename << "final_background_" << nCellsPerDirection;
        for (std::size_t i = 0; i < nSolvedCells; ++i) {
            const std::size_t cellRawId = solvedCellRawIds[i];

            for (int k = 0; k < N_FIELDS; ++k) {
                conservatives[k] = cellConservatives[k].rawAt(cellRawId);
            }

            ::utils::conservative2primitive(conservatives.data(), primitives.data());

            for (int k = 0; k < N_FIELDS; ++k) {
                cellPrimitives[k].rawAt(cellRawId) = primitives[k];
            }
        }
        mesh.write(filename.str().c_str());
    }

    log::cout() << "Computation time (without disk saving time) is "
                << double(computeEnd - computeStart - diskTime) / CLOCKS_PER_SEC
                << std::endl;
    log::cout() << "Disk time "
                << double(diskTime) / CLOCKS_PER_SEC
                << std::endl;

    // Error check
    std::array<double, N_FIELDS> evalConservatives;

    if (nProcessors == 1) {
        nvtxRangePushA("ErrorEvaluation_HU");
        cellConservatives.cuda_updateHost();
        nvtxRangePop();
    }
    double error = 0.;
    for (std::size_t i = 0; i < nSolvedCells; ++i) {
        const std::size_t cellRawId = solvedCellRawIds[i];
        const Cell &cell = mesh.getCells().rawAt(cellRawId);

        std::array<double, N_FIELDS> conservatives;
        for (int k = 0; k < N_FIELDS; ++k) {
            conservatives[k] = cellConservatives[k].rawAt(cellRawId);
        }

        problem::evalCellExactConservatives(problemType, cell.getId(), computationInfo, tMax, evalConservatives.data());
        error += std::abs(conservatives[FID_P] - evalConservatives[FID_P]) * computationInfo.rawGetCellVolume(cellRawId);
    }

#if ENABLE_MPI
    if (mesh.isPartitioned()) {
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, mesh.getCommunicator());
    }
#endif

    log::cout() << std::endl;
    log::cout() << " ::::::::: Error check :::::::::" << std::endl;
    log::cout() << std::endl;
    log::cout() << " Final error:  " << std::setprecision(12) << std::scientific << error << std::endl;
    log::cout() << std::endl;

    // Clean-up
#if ENABLE_CUDA
    cellRHS.cuda_freeDevice();
    computationInfo.cuda_finalize();
    euler::cuda_finalize();
#endif

#if ENABLE_MPI
    // finalize Communicators
    if (mesh.isPartitioned()) {
        log::cout() << "Conservative cuda objects finalize" << std::endl;
        conservativeCommunicator->finalizeCudaObjects();
        log::cout() << "Conservative WORK cuda objects finalize" << std::endl;
        conservativeWorkCommunicator->finalizeCudaObjects();
        log::cout() << "Primitive cuda objects finalize" << std::endl;
        primitiveCommunicator->finalizeCudaObjects();
    }
#endif
}

int main(int argc, char *argv[])
{
    //
    // Initialization
    //

#if ENABLE_MPI
    // MPI Initialization
    MPI_Init(&argc, &argv);
#endif

    //
    // Computation
    //
    computation(argc, argv);

    //
    // Finalization
    //

#if ENABLE_MPI
    // MPI finalization
    MPI_Finalize();
#endif
}
