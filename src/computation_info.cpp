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

using namespace bitpit;

/*!
 * \brief The ComputationInfo class provides an interface for defining
 * computation info.
 */

/*!
 * Creates a new info.
 *
 * \param patch is patch from which the informations will be extracted
 */
ComputationInfo::ComputationInfo(VolumeKernel *patch)
    : MeshGeometricalInfo(patch, false)
{
    ComputationInfo::_init();

    extract();
}

/*!
 * Sets the patch associated to the info.
 */
void ComputationInfo::_init()
{
    MeshGeometricalInfo::_init();

    m_cellSolveMethods.setStaticKernel(&m_volumePatch->getCells());

    m_interfaceSolveMethods.setStaticKernel(&m_volumePatch->getInterfaces());
}

/*!
 * Internal function to reset the information.
 */
void ComputationInfo::_reset()
{
    MeshGeometricalInfo::_reset();
}

/*!
 * Internal function to extract global information from the patch.
 */
void ComputationInfo::_extract()
{
    // Extract mesh information
    MeshGeometricalInfo::_extract();

    // Count solved cells and initialize solve method
    std::size_t nSolvedCells = 0;
    for (VolumeKernel::CellConstIterator cellItr = m_patch->cellConstBegin(); cellItr != m_patch->cellConstEnd(); ++cellItr) {
        std::size_t cellRawId = cellItr.getRawIndex();
        const Cell &cell = *cellItr;
        const std::array<float, 3> &cellCentroid = rawGetCellCentroid(cellRawId);

        // Identify solve method
        bool isSolved = body::isPointFluid(cellCentroid);
#if ENABLE_MPI
        if (isSolved) {
            isSolved = cell.isInterior();
        }
#endif
        m_cellSolveMethods.rawSet(cellRawId, (isSolved ? 1 : 0));

        // Count solved cells
        if (isSolved) {
            ++nSolvedCells;
        }
    }

    // Identify solved cells
    m_solvedCellRawIds.reserve(nSolvedCells);
    for (VolumeKernel::CellConstIterator cellItr = m_patch->cellConstBegin(); cellItr != m_patch->cellConstEnd(); ++cellItr) {
        std::size_t cellRawId = cellItr.getRawIndex();
        if (!m_cellSolveMethods.rawAt(cellRawId)) {
            continue;
        }

        m_solvedCellRawIds.push_back(cellRawId);
    }

    // Count solved interfaces and initialize solve method
    std::size_t nSolvedUniformInterfaces   = 0;
    std::size_t nSolvedBoundaryInterfaces = 0;
    for (VolumeKernel::InterfaceConstIterator interfaceItr = m_patch->interfaceConstBegin(); interfaceItr != m_patch->interfaceConstEnd(); ++interfaceItr) {
        std::size_t interfaceRawId = interfaceItr.getRawIndex();
        const Interface &interface = *interfaceItr;

        long ownerId = interface.getOwner();
        long neighId = interface.getNeigh();

        // Identify solve method
        int solveMethod = m_cellSolveMethods.at(ownerId);
        if (neighId >= 0) {
            solveMethod = std::max(m_cellSolveMethods.at(neighId), solveMethod);
        }
        m_interfaceSolveMethods.rawAt(interfaceRawId) = solveMethod;

        // Count solved interfaces
        if (solveMethod != 0) {
            if (!isInterfaceBoundary(interface)) {
                ++nSolvedUniformInterfaces;
            } else {
                ++nSolvedBoundaryInterfaces;
            }
        }
    }

    // Evaluate interface data
    m_solvedUniformInterfaceRawIds.reserve(nSolvedUniformInterfaces);
    m_solvedUniformInterfaceOwnerRawIds.reserve(nSolvedUniformInterfaces);
    m_solvedUniformInterfaceNeighRawIds.reserve(nSolvedUniformInterfaces);
    m_solvedBoundaryInterfaceRawIds.reserve(nSolvedBoundaryInterfaces);
    m_solvedBoundaryInterfaceSigns.reserve(nSolvedBoundaryInterfaces);
    m_solvedBoundaryInterfaceFluidRawIds.reserve(nSolvedBoundaryInterfaces);
    for (VolumeKernel::InterfaceConstIterator interfaceItr = m_patch->interfaceConstBegin(); interfaceItr != m_patch->interfaceConstEnd(); ++interfaceItr) {
        std::size_t interfaceRawId = interfaceItr.getRawIndex();
        if (!m_interfaceSolveMethods.rawAt(interfaceRawId)) {
            continue;
        }

        const Interface &interface = *interfaceItr;

        long ownerId = interface.getOwner();
        VolumeKernel::CellConstIterator ownerItr = m_patch->getCellConstIterator(ownerId);
        std::size_t ownerRawId = ownerItr.getRawIndex();

        long neighId = interface.getNeigh();
        std::size_t neighRawId;
        if (neighId >= 0) {
            VolumeKernel::CellConstIterator neighItr = m_patch->getCellConstIterator(neighId);
            neighRawId = neighItr.getRawIndex();
        }

        if (!isInterfaceBoundary(interface)) {
            m_solvedUniformInterfaceRawIds.push_back(interfaceRawId);
            m_solvedUniformInterfaceOwnerRawIds.push_back(ownerRawId);
            m_solvedUniformInterfaceNeighRawIds.push_back(neighRawId);
        } else {
            int boundarySign = 1;
            std::size_t fluidCellRawId = ownerRawId;
            if (!m_cellSolveMethods.at(ownerId)) {
                boundarySign = -1;
                fluidCellRawId = neighRawId;
            }

            m_solvedBoundaryInterfaceRawIds.push_back(interfaceRawId);
            m_solvedBoundaryInterfaceSigns.push_back(boundarySign);
            m_solvedBoundaryInterfaceFluidRawIds.push_back(fluidCellRawId);

        }
    }

    // Initialize storage for reconstructions
    m_solvedInterfaceLeftReconstructions.resize(N_FIELDS * std::max(nSolvedUniformInterfaces, nSolvedBoundaryInterfaces));
    m_solvedInterfaceRightReconstructions.resize(N_FIELDS * std::max(nSolvedUniformInterfaces, nSolvedBoundaryInterfaces));
}

/*!
 * Gheck if the interface is a boundary.
 *
 * \param interface is the interface
 * \result Returns true if the interface is a boundary, false otherwise.
 */
bool ComputationInfo::isInterfaceBoundary(const Interface &interface) const
{
    long neighId = interface.getNeigh();
    if (neighId < 0) {
        return true;
    }

    long ownerId = interface.getOwner();

    const std::array<float, 3> &cellCentroid  = getCellCentroid(ownerId);
    const std::array<float, 3> &neighCentroid = getCellCentroid(neighId);

    bool isOwnerFluid = body::isPointFluid(cellCentroid);
    bool isNeighFluid = body::isPointFluid(neighCentroid);

    return (isOwnerFluid != isNeighFluid);
}

/*!
 * Gets cells solve method.
 *
 * \result Cells solve method.
 */
const ScalarPiercedStorage<int> & ComputationInfo::getCellSolveMethods() const
{
    return m_cellSolveMethods;
}

/*!
 * Gets the list of solved cells raw ids.
 *
 * \result The list of solved cells raw ids.
 */
const ScalarStorage<std::size_t> & ComputationInfo::getSolvedCellRawIds() const
{
    return m_solvedCellRawIds;
}

/*!
 * Gets interfaces solve method.
 *
 * \result Interfaces solve method.
 */
const ScalarPiercedStorage<int> & ComputationInfo::getInterfaceSolveMethods() const
{
    return m_interfaceSolveMethods;
}

/*!
 * Gets the list of solved uniform interfaces raw ids.
 *
 * \result The list of solved uniform interfaces raw ids.
 */
const ScalarStorage<std::size_t> & ComputationInfo::getSolvedUniformInterfaceRawIds() const
{
    return m_solvedUniformInterfaceRawIds;
}

/*!
 * Gets the list of solved uniform interfaces owner raw ids.
 *
 * \result The list of solved uniform interfaces owner raw ids.
 */
const ScalarStorage<std::size_t> & ComputationInfo::getSolvedUniformInterfaceOwnerRawIds() const
{
    return m_solvedUniformInterfaceOwnerRawIds;
}

/*!
 * Gets the list of uniform interfaces neigh raw ids.
 *
 * \result The list of uniform interfaces neigh raw ids.
 */
const ScalarStorage<std::size_t> & ComputationInfo::getSolvedUniformInterfaceNeighRawIds() const
{
    return m_solvedUniformInterfaceNeighRawIds;
}

/*!
 * Gets the list of solved boundary interfaces raw ids.
 *
 * \result The list of solved boundary interfaces raw ids.
 */
const ScalarStorage<std::size_t> & ComputationInfo::getSolvedBoundaryInterfaceRawIds() const
{
    return m_solvedBoundaryInterfaceRawIds;
}

/*!
 * Gets the list of solved boundary interfaces signs.
 *
 * \result The list of solved boundary interfaces signs.
 */
const ScalarStorage<std::size_t> & ComputationInfo::getSolvedBoundaryInterfaceSigns() const
{
    return m_solvedBoundaryInterfaceSigns;
}

/*!
 * Gets the list of boundary interfaces fluid raw ids.
 *
 * \result The list of boundary interfaces fluid raw ids.
 */
const ScalarStorage<std::size_t> & ComputationInfo::getSolvedBoundaryInterfaceFluidRawIds() const
{
    return m_solvedBoundaryInterfaceFluidRawIds;
}

/*!
 * Gets a reference to the left interface reconstructions storage.
 *
 * \result A reference to the left interface reconstructions storage.
 */
ScalarStorage<float> & ComputationInfo::getSolvedInterfaceLeftReconstructions()
{
    return m_solvedInterfaceLeftReconstructions;
}

/*!
 * Gets a constant reference to the left interface reconstructions storage.
 *
 * \result A constant reference to the left interface reconstructions storage.
 */
const ScalarStorage<float> & ComputationInfo::getSolvedInterfaceLeftReconstructions() const
{
    return m_solvedInterfaceLeftReconstructions;
}

/*!
 * Gets a reference to the right interface reconstructions storage.
 *
 * \result A reference to the right interface reconstructions storage.
 */
ScalarStorage<float> & ComputationInfo::getSolvedInterfaceRightReconstructions()
{
    return m_solvedInterfaceRightReconstructions;
}

/*!
 * Gets a constant reference to the right interface reconstructions storage.
 *
 * \result A constant reference to the right interface reconstructions storage.
 */
const ScalarStorage<float> & ComputationInfo::getSolvedInterfaceRightReconstructions() const
{
    return m_solvedInterfaceRightReconstructions;
}
