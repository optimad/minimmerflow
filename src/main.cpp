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
#include "constants.hpp"
#include "euler.hpp"
#include "mesh_info.hpp"
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
    log::cout().setDefaultVisibility(log::GLOBAL);
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

    std::size_t haloSize = order;

    log::cout() << "*** Calling VolOctree constructor with " << nCellsPerDirection
                << " cells per direction, which will be uniformly refined " <<  initialRefs
                << " times." << std::endl;

#if ENABLE_MPI
    VolOctree mesh(dimensions, origin, length, length / nCellsPerDirection, MPI_COMM_WORLD, haloSize);
#else
    VolOctree mesh(dimensions, origin, length, length / nCellsPerDirection, haloSize);
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

    // Initialize mesh data
    log::cout() << std::endl;
    log::cout() << "Mesh data initialization..."  << std::endl;

    MeshGeometricalInfo meshInfo(&mesh);

    const std::vector<std::size_t> &cellRawIds = meshInfo.getCellRawIds();
    const std::size_t nCells = cellRawIds.size();

    const std::vector<std::size_t> &internalCellRawIds = meshInfo.getInternalCellRawIds();
    const std::size_t nInternalCells = internalCellRawIds.size();

    const std::vector<std::size_t> &interfaceRawIds = meshInfo.getInterfaceRawIds();
    const std::size_t nInterfaces = interfaceRawIds.size();

    log_memory_status();

    // Initialize cell storage
    log::cout() << std::endl;
    log::cout() << "Storage initialization..."  << std::endl;

    CellStorageBool cellSolvedFlag(1, &mesh.getCells());
    CellStorageBool cellFluidFlag(1, &mesh.getCells());
    CellStorageDouble cellPrimitives(N_FIELDS, &mesh.getCells());
    CellStorageDouble cellConservatives(N_FIELDS, &mesh.getCells());
    CellStorageDouble cellConservativesWork(N_FIELDS, &mesh.getCells());
    CellStorageDouble cellRHS(N_FIELDS, &mesh.getCells());

    // Initialize fluid and solved flag
    for (std::size_t i = 0; i < nCells; ++i) {
        const std::size_t cellRawId = cellRawIds[i];
        const Cell &cell = mesh.getCells().rawAt(cellRawId);
        const std::array<double, 3> &cellCentroid = meshInfo.rawGetCellCentroid(cellRawId);

        bool isFluid = body::isPointFluid(cellCentroid);
        cellFluidFlag.rawSet(cellRawId, isFluid);

        bool isSolved = isFluid;
#if ENABLE_MPI
        if (isSolved) {
            isSolved = cell.isInterior();
        }
#endif
        cellSolvedFlag.rawSet(cellRawId, isSolved);
    }
    log_memory_status();

    // Initialize reconstruction
    log::cout() << std::endl;
    log::cout() << "Reconstruction initialization..."  << std::endl;

    ReconstructionCalculator reconstructionCalculator(meshInfo, order);

    log_memory_status();

    // Boundary conditions
    log::cout() << std::endl;
    log::cout() << "Boundary conditions initialization..."  << std::endl;

    InterfaceStorageInt interfaceBCs(1, &mesh.getInterfaces());
    for (std::size_t i = 0; i < nInterfaces; ++i) {
        const std::size_t interfaceRawId = interfaceRawIds[i];
        const Interface &interface = mesh.getInterfaces().rawAt(interfaceRawId);
        long interfaceId = interface.getId();

        bool isIntrBorder = interface.isBorder();
        if (isIntrBorder) {
            interfaceBCs[interfaceId] = problem::getBorderBCType(problemType, interfaceId, meshInfo);
        } else {
            const long ownerId = interface.getOwner();
            VolumeKernel::CellConstIterator ownerItr = mesh.getCellConstIterator(ownerId);
            const std::size_t ownerRawId = ownerItr.getRawIndex();
            const bool ownerIsFluid = cellFluidFlag.rawAt(ownerRawId);

            const long neighId = interface.getNeigh();
            VolumeKernel::CellConstIterator neighItr = mesh.getCellConstIterator(neighId);
            const std::size_t neighRawId = neighItr.getRawIndex();
            const bool neighIsFluid = cellFluidFlag.rawAt(neighRawId);

            if (ownerIsFluid ^ neighIsFluid) {
                interfaceBCs[interfaceId] = BC_WALL;
            } else {
                interfaceBCs[interfaceId] = BC_NONE;
            }
        }
    }
    log_memory_status();

    // Initialize output
    log::cout() << std::endl;
    log::cout() << "Output initialization..."  << std::endl;

    SolverWriter solverStreamer(&mesh, &cellPrimitives, &cellConservatives, &cellRHS, &cellSolvedFlag);

    VTKUnstructuredGrid &vtk = mesh.getVTK();
    vtk.setCounter(0);

    vtk.addData<int>("solved"        , VTKFieldType::SCALAR, VTKLocation::CELL, &solverStreamer);
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

    std::unique_ptr<GhostCommunicator> conservativeCommunicator;
    std::unique_ptr<GhostCommunicator> conservativeWorkCommunicator;

    std::unique_ptr<CellBufferStreamer> conservativeGhostStreamer;
    std::unique_ptr<CellBufferStreamer> conservativeWorkGhostStreamer;
    if (mesh.isPartitioned()) {
        // Conservative fields
        conservativeCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh));
        conservativeCommunicator->resetExchangeLists();
        conservativeCommunicator->setRecvsContinuous(true);

        conservativeGhostStreamer = std::unique_ptr<CellBufferStreamer>(new CellBufferStreamer(&cellConservatives));
        conservativeCommunicator->addData(conservativeGhostStreamer.get());

        // Conservative tields tmp
        conservativeWorkCommunicator = std::unique_ptr<GhostCommunicator>(new GhostCommunicator(&mesh));
        conservativeWorkCommunicator->resetExchangeLists();
        conservativeWorkCommunicator->setRecvsContinuous(true);

        conservativeWorkGhostStreamer = std::unique_ptr<CellBufferStreamer>(new CellBufferStreamer(&cellConservativesWork));
        conservativeWorkCommunicator->addData(conservativeWorkGhostStreamer.get());
    }
#endif

    // Initial conditions
    log::cout() << std::endl;
    log::cout() << "Initial conditions evaluation..."  << std::endl;

    for (std::size_t i = 0; i < nCells; ++i) {
        const std::size_t cellRawId = cellRawIds[i];
        const Cell &cell = mesh.getCells().rawAt(cellRawId);

        double *conservatives = cellConservatives.rawData(cellRawId);
        problem::evalCellInitalConservatives(problemType, cell, meshInfo, conservatives);

        double *primitives = cellPrimitives.rawData(cellRawId);
        ::utils::conservative2primitive(conservatives, primitives);
    }

    mesh.write();

    log_memory_status();

    // Find smallest cell
    double minCellSize = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < nInternalCells; ++i) {
        const std::size_t cellRawId = internalCellRawIds[i];

        minCellSize = std::min(meshInfo.rawGetCellSize(cellRawId), minCellSize);
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

        //
        // FIRST RK STAGE
        //

        // Update reconstruction calculator
        reconstructionCalculator.update(cellConservatives);

        // Compute the residuals
        euler::computeRHS(problemType, meshInfo, cellSolvedFlag, reconstructionCalculator, interfaceBCs, &cellRHS, &maxEig);

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
        for (std::size_t i = 0; i < nInternalCells; ++i) {
            const std::size_t cellRawId = internalCellRawIds[i];
            bool isCellSolved = cellSolvedFlag.rawAt(cellRawId);
            if (!isCellSolved) {
                continue;
            }

            const double cellVolume = meshInfo.rawGetCellVolume(cellRawId);
            const double *RHS = cellRHS.rawData(cellRawId);
            const double *conservative = cellConservatives.rawData(cellRawId);
            double *conservativeTmp = cellConservativesWork.rawData(cellRawId);
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

        // Update reconstruction calculator
        reconstructionCalculator.update(cellConservativesWork);

        // Compute residuals
        euler::computeRHS(problemType, meshInfo, cellSolvedFlag, reconstructionCalculator, interfaceBCs, &cellRHS, &maxEig);
#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            MPI_Allreduce(MPI_IN_PLACE, &maxEig, 1, MPI_DOUBLE, MPI_MAX, mesh.getCommunicator());
        }
#endif

        log::cout() << "(maxEig after second stage = " << maxEig << ")" << std::endl;

        //
        // THIRD RK STAGE
        //
        for (std::size_t i = 0; i < nInternalCells; ++i) {
            const std::size_t cellRawId = internalCellRawIds[i];
            bool isCellSolved = cellSolvedFlag.rawAt(cellRawId);
            if (!isCellSolved) {
                continue;
            }

            double cellVolume = meshInfo.rawGetCellVolume(cellRawId);
            const double *RHS = cellRHS.rawData(cellRawId);
            const double *conservative = cellConservatives.rawData(cellRawId);
            double *conservativeTmp = cellConservativesWork.rawData(cellRawId);
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

        // Update reconstruction calculator
        reconstructionCalculator.update(cellConservativesWork);

        // Compute residuals
        euler::computeRHS(problemType, meshInfo, cellSolvedFlag, reconstructionCalculator, interfaceBCs, &cellRHS, &maxEig);
#if ENABLE_MPI
        if (mesh.isPartitioned()) {
            MPI_Allreduce(MPI_IN_PLACE, &maxEig, 1, MPI_DOUBLE, MPI_MAX, mesh.getCommunicator());
        }
#endif

        log::cout() << "(maxEig after third stage = " << maxEig << ")" << std::endl;

        //
        // CLOSE RK STEP
        //
        for (std::size_t i = 0; i < nInternalCells; ++i) {
            const std::size_t cellRawId = internalCellRawIds[i];
            bool isCellSolved = cellSolvedFlag.rawAt(cellRawId);
            if (!isCellSolved) {
                continue;
            }

            const double cellVolume = meshInfo.rawGetCellVolume(cellRawId);
            const double *RHS = cellRHS.rawData(cellRawId);
            double *conservative = cellConservatives.rawData(cellRawId);
            const double *conservativeTmp = cellConservativesWork.rawData(cellRawId);
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

        // Update timestep information
        t +=dt;
        step++;

        // Write the solution
        if (t > nextSave){
          clock_t diskStart = clock();
          for (std::size_t i = 0; i < nCells; ++i) {
              const std::size_t cellRawId = cellRawIds[i];
              const long cellId = mesh.getCells().rawFind(cellRawId).getId();

              const double *conservative = cellConservatives.data(cellId);
              double *primitives = cellPrimitives.data(cellId);
              ::utils::conservative2primitive(conservative, primitives);
          }
          mesh.write();

          diskTime += clock() - diskStart;
          nextSave += (tMax - tMin) / nSaves;
        }
    }
    clock_t computeEnd = clock();

    // Save final data
    {
        std::stringstream filename;
        filename << "final_background_" << nCellsPerDirection;
        for (std::size_t i = 0; i < nCells; ++i) {
            const std::size_t cellRawId = cellRawIds[i];
            const long cellId = mesh.getCells().rawFind(cellRawId).getId();

            const double *conservative = cellConservatives.data(cellId);
            double *primitives = cellPrimitives.data(cellId);
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
    for (std::size_t i = 0; i < nInternalCells; ++i) {
        const std::size_t cellRawId = internalCellRawIds[i];
        const Cell &cell = mesh.getCells().rawAt(cellRawId);
        const long cellId = cell.getId();

        const double *conservatives = cellConservatives.data(cellId);
        problem::evalCellExactConservatives(problemType, cell, meshInfo, tMax, evalConservatives.data());
        error += std::abs(conservatives[FID_P] - evalConservatives[FID_P]) * meshInfo.getCellVolume(cellId);
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
