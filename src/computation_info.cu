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
    m_cellSolveMethods.cuda_allocate();

    m_solvedCellRawIds.cuda_allocate();

    m_solvedUniformInterfaceRawIds.cuda_allocate();
    m_solvedUniformInterfaceOwnerRawIds.cuda_allocate();
    m_solvedUniformInterfaceNeighRawIds.cuda_allocate();

    m_solvedBoundaryInterfaceRawIds.cuda_allocate();
    m_solvedBoundaryInterfaceSigns.cuda_allocate();
    m_solvedBoundaryInterfaceFluidRawIds.cuda_allocate();

    // Copy data to the device
    m_cellSolveMethods.cuda_updateDevice();

    m_solvedCellRawIds.cuda_updateDevice();

    m_solvedUniformInterfaceRawIds.cuda_updateDevice();
    m_solvedUniformInterfaceOwnerRawIds.cuda_updateDevice();
    m_solvedUniformInterfaceNeighRawIds.cuda_updateDevice();

    m_solvedBoundaryInterfaceRawIds.cuda_updateDevice();
    m_solvedBoundaryInterfaceSigns.cuda_updateDevice();
    m_solvedBoundaryInterfaceFluidRawIds.cuda_updateDevice();
}

/*!
 * Finalize CUDA operations.
 */
void ComputationInfo::cuda_finalize()
{
    // Finalize mesh operations
    MeshGeometricalInfo::cuda_finalize();

    // Deallocate device memory
    m_cellSolveMethods.cuda_free();

    m_solvedCellRawIds.cuda_free();

    m_solvedUniformInterfaceRawIds.cuda_free();
    m_solvedUniformInterfaceOwnerRawIds.cuda_free();
    m_solvedUniformInterfaceNeighRawIds.cuda_free();

    m_solvedBoundaryInterfaceRawIds.cuda_free();
    m_solvedBoundaryInterfaceSigns.cuda_free();
    m_solvedBoundaryInterfaceFluidRawIds.cuda_free();
}

/*!
 * Gets a constant pointer to the device storage for the cell solve method.
 *
 * \result A constant pointer to the device storage for the cell solve method.
 */
const int * ComputationInfo::cuda_getCellSolveMethodDevData() const
{
    return m_cellSolveMethods.cuda_devData();
}

/*!
 * Gets a constant pointer to the device storage for the solved cell raw ids.
 *
 * \result A constant pointer to the device storage for the solved cell raw ids.
 */
const std::size_t * ComputationInfo::cuda_getSolvedCellRawIdDevData() const
{
    return m_solvedCellRawIds.cuda_devData();
}

/*!
 * Gets a constant pointer to the device storage for the interface solve method.
 *
 * \result A constant pointer to the device storage for the interface solve
 * method.
 */
const int * ComputationInfo::cuda_getInterfaceSolveMethodDevData() const
{
    return m_interfaceSolveMethods.cuda_devData();
}

/*!
 * Gets a constant pointer to the device storage for the solved uniform
 * interface raw ids.
 *
 * \result A constant pointer to the device storage for the solved uniform
 * interface raw ids.
 */
const std::size_t * ComputationInfo::cuda_getSolvedUniformInterfaceRawIdDevData() const
{
    return m_solvedUniformInterfaceRawIds.cuda_devData();
}

/*!
 * Gets a constant pointer to the device storage for the solved uniform
 * interface owner raw ids.
 *
 * \result A constant pointer to the device storage for the solved uniform
 * interface owner raw ids.
 */
const std::size_t * ComputationInfo::cuda_getSolvedUniformInterfaceOwnerRawIdDevData() const
{
    return m_solvedUniformInterfaceOwnerRawIds.cuda_devData();
}

/*!
 * Gets a constant pointer to the device storage for the solved uniform
 * interface neighbour raw ids.
 *
 * \result A constant pointer to the device storage for the solved uniform
 * interface neighbour raw ids.
 */
const std::size_t * ComputationInfo::cuda_getSolvedUniformInterfaceNeighRawIdDevData() const
{
    return m_solvedUniformInterfaceNeighRawIds.cuda_devData();
}

/*!
 * Gets a constant pointer to the device storage for the solved boundary
 * interface raw ids.
 *
 * \result A constant pointer to the device storage for the solved boundary
 * interface raw ids.
 */
const std::size_t * ComputationInfo::cuda_getSolvedBoundaryInterfaceRawIdDevData() const
{
    return m_solvedBoundaryInterfaceRawIds.cuda_devData();
}

/*!
 * Gets a constant pointer to the device storage for the solved boundary
 * interface signs.
 *
 * \result A constant pointer to the device storage for the solved boundary
 * interface signs.
 */
const std::size_t * ComputationInfo::cuda_getSolvedBoundaryInterfaceSignDevData() const
{
    return m_solvedBoundaryInterfaceSigns.cuda_devData();
}

/*!
 * Gets a constant pointer to the device storage for the solved boundary
 * interface fluid raw ids.
 *
 * \result A constant pointer to the device storage for the solved boundary
 * interface fluid raw ids.
 */
const std::size_t * ComputationInfo::cuda_getSolvedBoundaryInterfaceFluidRawIdDevData() const
{
    return m_solvedBoundaryInterfaceFluidRawIds.cuda_devData();
}
