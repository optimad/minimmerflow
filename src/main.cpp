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
#include "utils.hpp"
#include "memory.hpp"

#include <bitpit_IO.hpp>
#include <bitpit_voloctree.hpp>

#if ENABLE_MPI
#include <mpi.h>
#endif
#include <vector>
#include <time.h>
#include <string>

#include <nvtx3/nvToolsExt.h>

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
    // Initialize constants
    constants::cuda_initialize();
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

    ScalarPiercedStorage<double> cellPrimitives(N_FIELDS, &mesh.getCells());
    ScalarPiercedStorage<double> cellConservatives(N_FIELDS, &mesh.getCells());
    ScalarPiercedStorage<double> cellConservativesWork(N_FIELDS, &mesh.getCells());
    ScalarPiercedStorage<double> cellRHS(N_FIELDS, &mesh.getCells());

#if ENABLE_CUDA
    cellRHS.cuda_allocateDevice();
    cellConservatives.cuda_allocateDevice();
    cellPrimitives.cuda_allocateDevice();
    cellConservativesWork.cuda_allocateDevice();
#endif

    // Set host storage pointers for OpenACC
    double *cellPrimitivesHostStorage = cellPrimitives.data();
    double *cellConservativesHostStorage = cellConservatives.data();
    double *cellConservativesWorkHostStorage = cellConservativesWork.data();
    double *cellRHSHostStorage = cellRHS.data();
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

#if ENABLE_MPI
    // Creating ghost communications for exchanging solved data
    log::cout() << std::endl;
    log::cout() << " * Inizializing ghost communications for exchanging ghost data" << std::endl;

    std::unique_ptr<GhostCommunicator> primitiveCommunicator;
    std::unique_ptr<GhostCommunicator> conservativeCommunicator;
    std::unique_ptr<GhostCommunicator> conservativeWorkCommunicator;

    std::unique_ptr<ValuePiercedStorageBufferStreamer<double>> primitiveGhostStreamer;
    std::unique_ptr<ValuePiercedStorageBufferStreamer<double>> conservativeGhostStreamer;
    std::unique_ptr<ValuePiercedStorageBufferStreamer<double>> conservativeWorkGhostStreamer;
    if (mesh.isPartitioned()) {
        // Primitive fields
        primitiveCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh));
        primitiveCommunicator->resetExchangeLists();
        primitiveCommunicator->setRecvsContinuous(true);

        primitiveGhostStreamer = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&cellConservatives));
        primitiveCommunicator->addData(primitiveGhostStreamer.get());

        // Conservative fields
        conservativeCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh));
        conservativeCommunicator->resetExchangeLists();
        conservativeCommunicator->setRecvsContinuous(true);

        conservativeGhostStreamer = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&cellConservatives));
        conservativeCommunicator->addData(conservativeGhostStreamer.get());

        // Conservative tields tmp
        conservativeWorkCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh));
        conservativeWorkCommunicator->resetExchangeLists();
        conservativeWorkCommunicator->setRecvsContinuous(true);

        conservativeWorkGhostStreamer = std::unique_ptr<ValuePiercedStorageBufferStreamer<double>>(new ValuePiercedStorageBufferStreamer<double>(&cellConservativesWork));
        conservativeWorkCommunicator->addData(conservativeWorkGhostStreamer.get());
    }
#endif

    // Initial conditions
    log::cout() << std::endl;
    log::cout() << "Initial conditions evaluation..."  << std::endl;

    for (std::size_t i = 0; i < nSolvedCells; ++i) {
        const std::size_t cellRawId = solvedCellRawIds[i];
        const long cellId = mesh.getCells().rawFind(cellRawId).getId();

        double *conservatives = cellConservatives.rawData(cellRawId);
        problem::evalCellInitalConservatives(problemType, cellId, computationInfo, conservatives);

        //double *primitives = cellPrimitives.rawData(cellRawId);
        //::utils::conservative2primitive(conservatives, primitives);
    }

    // UPdate conservative on gpu
#if ENABLE_CUDA
    cellConservatives.cuda_updateDevice();
#endif

#pragma acc parallel loop present(cellPrimitivesHostStorage, cellConservativesHostStorage, solvedCellRawIdsHostStorage)
     for (long i = 0; i < nSolvedCells; ++i) {
         double *primitives = &cellPrimitivesHostStorage[solvedCellRawIdsHostStorage[i]*N_FIELDS];
         const double *conservatives = &cellConservativesHostStorage[solvedCellRawIdsHostStorage[i]*N_FIELDS];
         ::utils::conservative2primitive(conservatives, primitives);
     }

     log::cout() << std::endl;
     log::cout() << "Update primitives on host..."  << std::endl;
#if ENABLE_CUDA
     cellPrimitives.cuda_updateHost();
#endif


#if ENABLE_MPI
    if (mesh.isPartitioned()) {
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
    while (t < tMax) {
        log::cout() << std::endl;
        log::cout() << "Step n. " << step << std::endl;

        double maxEig;

        std::string rangeName = "TimeStep" + std::to_string(step);
        nvtxRangePushA(rangeName.c_str());

        //
        // FIRST RK STAGE
        //

        // Compute the residuals
#if ENABLE_CUDA
        cellConservatives.cuda_updateDevice();
#endif

        reconstruction::computePolynomials(problemType, computationInfo, cellConservatives, solvedBoundaryInterfaceBCs);
        euler::computeRHS(problemType, computationInfo, order, solvedBoundaryInterfaceBCs, cellConservatives, &cellRHS, &maxEig);

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

        //
        // SECOND RK STAGE
        //
#pragma acc parallel loop present(cellVolumeHostStorage, cellConservativesHostStorage, solvedCellRawIdsHostStorage, cellRHSHostStorage, cellConservativesWorkHostStorage)
        for (std::size_t i = 0; i < nSolvedCells; ++i) {
            const std::size_t cellRawId = solvedCellRawIdsHostStorage[i];
            const double cellVolume = cellVolumeHostStorage[cellRawId];
            const double *RHS = &cellRHSHostStorage[cellRawId * N_FIELDS];
            const double *conservative = &cellConservativesHostStorage[cellRawId * N_FIELDS];
            double *conservativeTmp = &cellConservativesWorkHostStorage[cellRawId * N_FIELDS];
#pragma acc loop seq
            for (int k = 0; k < N_FIELDS; ++k) {
                conservativeTmp[k] = conservative[k] + dt * RHS[k] / cellVolume;
            }
        }

#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            conservativeWorkCommunicator->startAllExchanges();
            conservativeWorkCommunicator->completeAllExchanges();
        }
#endif

#if ENABLE_CUDA
        cellConservativesWork.cuda_updateHost();
#endif

        reconstruction::computePolynomials(problemType, computationInfo, cellConservativesWork, solvedBoundaryInterfaceBCs);
        euler::computeRHS(problemType, computationInfo, order, solvedBoundaryInterfaceBCs, cellConservativesWork, &cellRHS, &maxEig);
#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            MPI_Allreduce(MPI_IN_PLACE, &maxEig, 1, MPI_DOUBLE, MPI_MAX, mesh.getCommunicator());
        }
#endif

        log::cout() << "(maxEig after second stage = " << maxEig << ")" << std::endl;

        //
        // THIRD RK STAGE
        //
#pragma acc parallel loop present(cellVolumeHostStorage, cellConservativesHostStorage, solvedCellRawIdsHostStorage, cellRHSHostStorage, cellConservativesWorkHostStorage)
        for (std::size_t i = 0; i < nSolvedCells; ++i) {
            const std::size_t cellRawId = solvedCellRawIdsHostStorage[i];
            const double cellVolume = cellVolumeHostStorage[cellRawId];
            const double *RHS = &cellRHSHostStorage[cellRawId * N_FIELDS];
            const double *conservative = &cellConservativesHostStorage[cellRawId * N_FIELDS];
            double *conservativeTmp = &cellConservativesWorkHostStorage[cellRawId * N_FIELDS];
#pragma acc loop seq
            for (int k = 0; k < N_FIELDS; ++k) {
                conservativeTmp[k] = 0.75*conservative[k] + 0.25*(conservativeTmp[k] + dt * RHS[k] / cellVolume);
            }
        }

#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            conservativeWorkCommunicator->startAllExchanges();
            conservativeWorkCommunicator->completeAllExchanges();
        }
#endif

#if ENABLE_CUDA
        cellConservativesWork.cuda_updateHost();
#endif

        reconstruction::computePolynomials(problemType, computationInfo, cellConservativesWork, solvedBoundaryInterfaceBCs);
        euler::computeRHS(problemType, computationInfo, order, solvedBoundaryInterfaceBCs, cellConservativesWork, &cellRHS, &maxEig);
#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            MPI_Allreduce(MPI_IN_PLACE, &maxEig, 1, MPI_DOUBLE, MPI_MAX, mesh.getCommunicator());
        }
#endif

        log::cout() << "(maxEig after third stage = " << maxEig << ")" << std::endl;

        //
        // CLOSE RK STEP
        //
#pragma acc parallel loop present(cellVolumeHostStorage, cellConservativesHostStorage, solvedCellRawIdsHostStorage, cellRHSHostStorage, cellConservativesWorkHostStorage)
        for (std::size_t i = 0; i < nSolvedCells; ++i) {
            const std::size_t cellRawId = solvedCellRawIdsHostStorage[i];
            const double cellVolume = cellVolumeHostStorage[cellRawId];
            const double *RHS = &cellRHSHostStorage[cellRawId * N_FIELDS];
            double *conservative = &cellConservativesHostStorage[cellRawId * N_FIELDS];
            const double *conservativeTmp = &cellConservativesWorkHostStorage[cellRawId * N_FIELDS];
#pragma acc loop seq
            for (int k = 0; k < N_FIELDS; ++k) {
                conservative[k] = (1./3)*conservative[k] + (2./3)*(conservativeTmp[k] + dt * RHS[k] / cellVolume);
            }
        }

#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            conservativeCommunicator->startAllExchanges();
            conservativeCommunicator->completeAllExchanges();
        }
#endif

#if ENABLE_CUDA
        cellConservatives.cuda_updateHost();
#endif
        // Update timestep information
        t +=dt;
        step++;

        // Write the solution
        if (t > nextSave){
            clock_t diskStart = clock();
            for (std::size_t i = 0; i < nSolvedCells; ++i) {
                const std::size_t cellRawId = solvedCellRawIds[i];
                const double *conservative = cellConservatives.rawData(cellRawId);
                double *primitives = cellPrimitives.rawData(cellRawId);
                ::utils::conservative2primitive(conservative, primitives);
            }
            mesh.write();

            diskTime += clock() - diskStart;
            nextSave += (tMax - tMin) / nSaves;
        }
      
        nvtxRangePop();

    }
    clock_t computeEnd = clock();

    // Save final data
    {
        std::stringstream filename;
        filename << "final_background_" << nCellsPerDirection;
        for (std::size_t i = 0; i < nSolvedCells; ++i) {
            const std::size_t cellRawId = solvedCellRawIds[i];
            const double *conservative = cellConservatives.rawData(cellRawId);
            double *primitives = cellPrimitives.rawData(cellRawId);
            ::utils::conservative2primitive(conservative, primitives);
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

    double error = 0.;
    for (std::size_t i = 0; i < nSolvedCells; ++i) {
        const std::size_t cellRawId = solvedCellRawIds[i];
        const Cell &cell = mesh.getCells().rawAt(cellRawId);

        const double *conservatives = cellConservatives.rawData(cellRawId);
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
