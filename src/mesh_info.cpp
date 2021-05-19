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

#include "mesh_info.hpp"

using namespace bitpit;

/*!
 * \brief The MeshGeometricalInfo class provides an interface for defining patch info.
 */

/*!
 * Creates a new info.
 *
 * \param patch is patch from which the informations will be extracted
 */
MeshGeometricalInfo::MeshGeometricalInfo(VolumeKernel *patch)
    : PatchInfo(patch)
{
    m_volumePatch = dynamic_cast<const VolumeKernel *>(patch);
    if (!m_volumePatch) {
        throw std::runtime_error("Volume geometrical information can only be used with volume patches.");
    }

    MeshGeometricalInfo::_init();

    extract();
}

/*!
 * Get the patch.
 *
 * \result The patch.
 */
bitpit::VolumeKernel const & MeshGeometricalInfo::getPatch() const
{
    return *m_volumePatch;
}

/*!
 * Sets the patch associated to the info.
 */
void MeshGeometricalInfo::_init()
{
    m_cellVolumes.setStaticKernel(&m_volumePatch->getCells());
    m_cellSizes.setStaticKernel(&m_volumePatch->getCells());
    m_cellCentroids.setStaticKernel(&m_volumePatch->getCells());

    m_interfaceAreas.setStaticKernel(&m_volumePatch->getInterfaces());
    m_interfaceCentroids.setStaticKernel(&m_volumePatch->getInterfaces());
    m_interfaceNormals.setStaticKernel(&m_volumePatch->getInterfaces());
    m_interfaceTangents.setStaticKernel(&m_volumePatch->getInterfaces());
}

/*!
 * Internal function to reset the information.
 */
void MeshGeometricalInfo::_reset()
{
}

/*!
 * Internal function to extract global information from the patch.
 */
void MeshGeometricalInfo::_extract()
{
    // Evaluate cell data
    for (VolumeKernel::CellConstIterator cellItr = m_patch->cellConstBegin(); cellItr != m_patch->cellConstEnd(); ++cellItr) {
        long cellId = cellItr.getId();
        std::size_t cellRawId = cellItr.getRawIndex();

        m_cellVolumes.rawSet(cellRawId, m_volumePatch->evalCellVolume(cellId));
        m_cellSizes.rawSet(cellRawId, m_volumePatch->evalCellSize(cellId));
        m_cellCentroids.rawSet(cellRawId, m_volumePatch->evalCellCentroid(cellId));
    }

    // Evaluate interface data
    for (VolumeKernel::InterfaceConstIterator interfaceItr = m_patch->interfaceConstBegin(); interfaceItr != m_patch->interfaceConstEnd(); ++interfaceItr) {
        long cellId = interfaceItr.getId();
        std::size_t interfaceRawId = interfaceItr.getRawIndex();

        m_interfaceAreas.rawSet(interfaceRawId, m_volumePatch->evalInterfaceArea(cellId));
        m_interfaceCentroids.rawSet(interfaceRawId, m_volumePatch->evalInterfaceCentroid(cellId));
        m_interfaceNormals.rawSet(interfaceRawId, m_volumePatch->evalInterfaceNormal(cellId));
        m_interfaceTangents.rawSet(interfaceRawId, {{1., 0, 0}});
    }
}

/*!
 * Gets the dimension of the patch.
 *
 * \result The dimension of the patch.
 */
int MeshGeometricalInfo::getDimension() const
{
    return m_patch->getDimension();
}

/*!
 * Gets the volume of the specified cell.
 *
 * \param id is the id of the cell
 * \result The volume of the specified cell.
 */
double MeshGeometricalInfo::getCellVolume(long id) const
{
    return m_cellVolumes.at(id);
}

/*!
 * Gets the volume of the cell at the specified raw position.
 *
 * \param pos is the raw position of the item
 * \result The volume of the specified cell.
 */
double MeshGeometricalInfo::rawGetCellVolume(size_t pos) const
{
    return m_cellVolumes.rawAt(pos);
}

/*!
 * Gets a constant reference to the cell volume storage.
 *
 * \result A constant reference to the cell volume storage.
 */
const PiercedStorage<double, long> & MeshGeometricalInfo::getCellVolumes() const
{
    return m_cellVolumes;
}

/*!
 * Gets a reference to the cell volume storage.
 *
 * \result A reference to the cell volume storage.
 */
PiercedStorage<double, long> & MeshGeometricalInfo::getCellVolumes()
{
    return m_cellVolumes;
}

/*!
 * Gets the size of the specified cell.
 *
 * \param id is the id of the cell
 * \result The size of the specified cell.
 */
double MeshGeometricalInfo::getCellSize(long id) const
{
    return m_cellSizes.at(id);
}

/*!
 * Gets the size of the cell at the specified raw position.
 *
 * \param pos is the raw position of the item
 * \result The size of the specified cell.
 */
double MeshGeometricalInfo::rawGetCellSize(size_t pos) const
{
    return m_cellSizes.rawAt(pos);
}

/*!
 * Gets a constant reference to the cell size storage.
 *
 * \result A constant reference to the cell size storage.
 */
const PiercedStorage<double, long> & MeshGeometricalInfo::getCellSizes() const
{
    return m_cellSizes;
}

/*!
 * Gets a reference to the cell size storage.
 *
 * \result A reference to the cell size storage.
 */
PiercedStorage<double, long> & MeshGeometricalInfo::getCellSizes()
{
    return m_cellSizes;
}

/*!
 * Gets the centroid of the specified cell.
 *
 * \param id is the id of the cell
 * \result The centroid of the specified cell.
 */
const std::array<double, 3> & MeshGeometricalInfo::getCellCentroid(long id) const
{
    return m_cellCentroids.at(id);
}

/*!
 * Gets the centroid of the cell at the specified raw position.
 *
 * \param pos is the raw position of the item
 * \result The centroid of the specified cell.
 */
const std::array<double, 3> & MeshGeometricalInfo::rawGetCellCentroid(size_t pos) const
{
    return m_cellCentroids.rawAt(pos);
}

/*!
 * Gets a constant reference to the cell centroid storage.
 *
 * \result A constant reference to the cell centroid storage.
 */
const PiercedStorage<std::array<double, 3>, long> & MeshGeometricalInfo::getCellCentroids() const
{
    return m_cellCentroids;
}

/*!
 * Gets a reference to the cell centroid storage.
 *
 * \result A reference to the cell centroid storage.
 */
PiercedStorage<std::array<double, 3>, long> & MeshGeometricalInfo::getCellCentroids()
{
    return m_cellCentroids;
}


/*!
 * Gets the area of the specified interface.
 *
 * \param id is the id of the interface
 * \result The centroid of the specified interface.
 */
double MeshGeometricalInfo::getInterfaceArea(long id) const
{
    return m_interfaceAreas.at(id);
}

/*!
 * Gets the area of the interface at the specified raw position.
 *
 * \param pos is the raw position of the item
 * \result The area of the specified interface.
 */
double MeshGeometricalInfo::rawGetInterfaceArea(size_t pos) const
{
    return m_interfaceAreas.rawAt(pos);
}

/*!
 * Gets a constant reference to the interface area storage.
 *
 * \result A constant reference to the interface area storage.
 */
const PiercedStorage<double, long> & MeshGeometricalInfo::getInterfaceAreas() const
{
    return m_interfaceAreas;
}

/*!
 * Gets a reference to the interface area storage.
 *
 * \result A reference to the interface area storage.
 */
PiercedStorage<double, long> & MeshGeometricalInfo::getInterfaceAreas()
{
    return m_interfaceAreas;
}

/*!
 * Gets the centroid of the specified interface.
 *
 * \param id is the id of the interface
 * \result The centroid of the specified interface.
 */
const std::array<double, 3> & MeshGeometricalInfo::getInterfaceCentroid(long id) const
{
    return m_interfaceCentroids.at(id);
}

/*!
 * Gets the centroid of the interface at the specified raw position.
 *
 * \param pos is the raw position of the item
 * \result The centroid of the specified interface.
 */
const std::array<double, 3> & MeshGeometricalInfo::rawGetInterfaceCentroid(size_t pos) const
{
    return m_interfaceCentroids.rawAt(pos);
}

/*!
 * Gets a constant reference to the interface centroid storage.
 *
 * \result A constant reference to the interface centroid storage.
 */
const PiercedStorage<std::array<double, 3>, long> & MeshGeometricalInfo::getInterfaceCentroids() const
{
    return m_interfaceCentroids;
}

/*!
 * Gets a reference to the interface centroid storage.
 *
 * \result A reference to the interface centroid storage.
 */
PiercedStorage<std::array<double, 3>, long> & MeshGeometricalInfo::getInterfaceCentroids()
{
    return m_interfaceCentroids;
}

/*!
 * Gets the normal of the specified interface.
 *
 * \param id is the id of the interface
 * \result The normal of the specified interface.
 */
const std::array<double, 3> & MeshGeometricalInfo::getInterfaceNormal(long id) const
{
    return m_interfaceNormals.at(id);
}

/*!
 * Gets the normal of the interface at the specified raw position.
 *
 * \param pos is the raw position of the item
 * \result The normal of the specified interface.
 */
const std::array<double, 3> & MeshGeometricalInfo::rawGetInterfaceNormal(size_t pos) const
{
    return m_interfaceNormals.rawAt(pos);
}

/*!
 * Gets a constant reference to the interface normal storage.
 *
 * \result A constant reference to the interface normal storage.
 */
const PiercedStorage<std::array<double, 3>, long> & MeshGeometricalInfo::getInterfaceNormals() const
{
    return m_interfaceNormals;
}

/*!
 * Gets a reference to the interface normal storage.
 *
 * \result A reference to the interface normal storage.
 */
PiercedStorage<std::array<double, 3>, long> & MeshGeometricalInfo::getInterfaceNormals()
{
    return m_interfaceNormals;
}

/*!
 * Gets the tangent of the specified interface.
 *
 * \param id is the id of the interface
 * \result The tangent of the specified interface.
 */
const std::array<double, 3> & MeshGeometricalInfo::getInterfaceTangent(long id) const
{
    return m_interfaceTangents.at(id);
}

/*!
 * Gets the tangent of the interface at the specified raw position.
 *
 * \param pos is the raw position of the item
 * \result The tangent of the specified interface.
 */
const std::array<double, 3> & MeshGeometricalInfo::rawGetInterfaceTangent(size_t pos) const
{
    return m_interfaceTangents.rawAt(pos);
}

/*!
 * Gets a constant reference to the interface tangent storage.
 *
 * \result A constant reference to the interface tangent storage.
 */
const PiercedStorage<std::array<double, 3>, long> & MeshGeometricalInfo::getInterfaceTangents() const
{
    return m_interfaceTangents;
}

/*!
 * Gets a reference to the interface tangent storage.
 *
 * \result A reference to the interface tangent storage.
 */
PiercedStorage<std::array<double, 3>, long> & MeshGeometricalInfo::getInterfaceTangents()
{
    return m_interfaceTangents;
}
