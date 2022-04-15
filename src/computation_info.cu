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

#include "computation_info.hpp"

/*!
 * Initialize CUDA operations.
 */
void ComputationInfo::cuda_initialize()
{
    // Initialize mesh operations
    MeshGeometricalInfo::cuda_initialize();

    // Allocate device memory
    m_cellSolveMethods.cuda_allocateDevice();

    m_solvedCellRawIds.cuda_allocateDevice();

    m_solvedUniformInterfaceRawIds.cuda_allocateDevice();
    m_solvedUniformInterfaceOwnerRawIds.cuda_allocateDevice();
    m_solvedUniformInterfaceNeighRawIds.cuda_allocateDevice();

    m_solvedBoundaryInterfaceRawIds.cuda_allocateDevice();
    m_solvedBoundaryInterfaceFluidRawIds.cuda_allocateDevice();
    m_solvedBoundaryInterfaceSigns.cuda_allocateDevice();

    // Copy data to the device
    m_cellSolveMethods.cuda_updateDevice();

    m_solvedCellRawIds.cuda_updateDevice();

    m_solvedUniformInterfaceRawIds.cuda_updateDevice();
    m_solvedUniformInterfaceOwnerRawIds.cuda_updateDevice();
    m_solvedUniformInterfaceNeighRawIds.cuda_updateDevice();

    m_solvedBoundaryInterfaceRawIds.cuda_updateDevice();
    m_solvedBoundaryInterfaceFluidRawIds.cuda_updateDevice();
    m_solvedBoundaryInterfaceSigns.cuda_updateDevice();
}

/*!
 * Finalize CUDA operations.
 */
void ComputationInfo::cuda_finalize()
{
    // Finalize mesh operations
    MeshGeometricalInfo::cuda_finalize();

    // Deallocate device memory
    m_cellSolveMethods.cuda_freeDevice();

    m_solvedCellRawIds.cuda_freeDevice();

    m_solvedUniformInterfaceRawIds.cuda_freeDevice();
    m_solvedUniformInterfaceOwnerRawIds.cuda_freeDevice();
    m_solvedUniformInterfaceNeighRawIds.cuda_freeDevice();

    m_solvedBoundaryInterfaceRawIds.cuda_freeDevice();
    m_solvedBoundaryInterfaceFluidRawIds.cuda_freeDevice();
    m_solvedBoundaryInterfaceSigns.cuda_freeDevice();
}

/*!
 * Resize GPU allocated memory
 */
void ComputationInfo::cuda_resize()
{
    // Initialize mesh operations
    MeshGeometricalInfo::cuda_resize();

    // Allocate device memory
    m_cellSolveMethods.cuda_resize(getSolvedCellRawIds().size());

    m_solvedCellRawIds.cuda_resize(getSolvedCellRawIds().size());

    m_solvedUniformInterfaceRawIds.cuda_resize(getSolvedUniformInterfaceRawIds().size());
    m_solvedUniformInterfaceOwnerRawIds.cuda_resize(getSolvedUniformInterfaceRawIds().size());
    m_solvedUniformInterfaceNeighRawIds.cuda_resize(getSolvedUniformInterfaceRawIds().size());

    m_solvedBoundaryInterfaceRawIds.cuda_resize(getSolvedBoundaryInterfaceRawIds().size());
    m_solvedBoundaryInterfaceFluidRawIds.cuda_resize(getSolvedBoundaryInterfaceRawIds().size());
    m_solvedBoundaryInterfaceSigns.cuda_resize(getSolvedBoundaryInterfaceRawIds().size());
}

