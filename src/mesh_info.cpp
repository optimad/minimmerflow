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
    : MeshGeometricalInfo(patch, true)
{
}

/*!
 * Creates a new info.
 *
 * \param patch is patch from which the informations will be extracted
 * \param extractInfo constrols if mesh information will be extracted
 */
MeshGeometricalInfo::MeshGeometricalInfo(VolumeKernel *patch, bool extractInfo)
    : PatchInfo(patch),
      m_cellCentroids(3, 1),
      m_interfaceCentroids(3, 1), m_interfaceNormals(3, 1), m_interfaceTangents(3, 1)
{
    m_volumePatch = dynamic_cast<const VolumeKernel *>(patch);
    if (!m_volumePatch) {
        throw std::runtime_error("Volume geometrical information can only be used with volume patches.");
    }

    MeshGeometricalInfo::_init();

    if (extractInfo) {
        extract();
    }
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
    for (int d = 0; d < 3; ++d) {
        m_cellCentroids[d].setStaticKernel(&m_volumePatch->getCells());
    }

    m_interfaceAreas.setStaticKernel(&m_volumePatch->getInterfaces());
    for (int d = 0; d < 3; ++d) {
        m_interfaceCentroids[d].setStaticKernel(&m_volumePatch->getInterfaces());
        m_interfaceNormals[d].setStaticKernel(&m_volumePatch->getInterfaces());
        m_interfaceTangents[d].setStaticKernel(&m_volumePatch->getInterfaces());
    }
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
        const std::array<double, 3> &cellCentroid = m_volumePatch->evalCellCentroid(cellId);
        for (int d = 0; d < 3; ++d) {
            m_cellCentroids[d].rawSet(cellRawId, cellCentroid[d]);
        }
    }

    // Evaluate interface data
    for (VolumeKernel::InterfaceConstIterator interfaceItr = m_patch->interfaceConstBegin(); interfaceItr != m_patch->interfaceConstEnd(); ++interfaceItr) {
        long interfaceId = interfaceItr.getId();
        std::size_t interfaceRawId = interfaceItr.getRawIndex();

        m_interfaceAreas.rawSet(interfaceRawId, m_volumePatch->evalInterfaceArea(interfaceId));

        const std::array<double, 3> &interfaceCentroid = m_volumePatch->evalInterfaceCentroid(interfaceId);
        const std::array<double, 3> &interfaceNormal   = m_volumePatch->evalInterfaceNormal(interfaceId);
        const std::array<double, 3> interfaceTangent   = {{1., 0, 0}};
        for (int d = 0; d < 3; ++d) {
            m_interfaceCentroids[d].rawSet(interfaceRawId, interfaceCentroid[d]);
            m_interfaceNormals[d].rawSet(interfaceRawId, interfaceNormal[d]);
            m_interfaceTangents[d].rawSet(interfaceRawId, interfaceTangent[d]);
        }
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
 * Gets the requested centroid component for the specified interface.
 *
 * \param id is the id of the cell
 * \param component is the requested component
 * \result The requested centroid component for the specified interface.
 */
double MeshGeometricalInfo::getCellCentroid(long id, int component) const
{
    return m_cellCentroids[component].at(id);
}

/*!
 * Gets the centroid of the given cell.
 *
 * \param id is the id of the cell
 * \param component is the requested component
 * \result The centroid of the specified cell.
 */
std::array<double, 3> MeshGeometricalInfo::getCellCentroid(long id) const
{
    std::size_t rawCellId = m_patch->getCells().find(id).getRawIndex();

    return rawGetCellCentroid(rawCellId);
}

/*!
 * Gets the requested centroid component for the specified interface.
 *
 * \param pos is the raw position of the item
 * \param component is the requested component
 * \result The requested centroid component for the specified interface.
 */
double MeshGeometricalInfo::rawGetCellCentroid(size_t pos, int component) const
{
    return m_cellCentroids[component].rawAt(pos);
}

/*!
 * Gets the centroid of the given cell.
 *
 * \param pos is the raw position of the item
 * \param component is the requested component
 * \result The centroid of the specified cell.
 */
std::array<double, 3> MeshGeometricalInfo::rawGetCellCentroid(size_t pos) const
{
    return {{m_cellCentroids[0].rawAt(pos), m_cellCentroids[1].rawAt(pos), m_cellCentroids[2].rawAt(pos)}};
}

/*!
 * Gets a constant reference to the cell centroid storage for the given
 * component.
 *
 * \param component is the requested component
 * \result A constant reference to the cell centroid storage.
 */
const bitpit::PiercedStorage<double, long> & MeshGeometricalInfo::getCellCentroids(int component) const
{
    return m_cellCentroids[component];
}

/*!
 * Gets a constant reference to the cell centroid storage for the given
 * component.
 *
 * \param component is the requested component
 * \result A reference to the cell centroid storage.
 */
bitpit::PiercedStorage<double, long> & MeshGeometricalInfo::getCellCentroids(int component)
{
    return m_cellCentroids[component];
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
 * Gets the requested centroid component for the specified interface.
 *
 * \param id is the id of the interface
 * \param component is the requested component
 * \result The requested centroid component for the specified interface.
 */
double MeshGeometricalInfo::getInterfaceCentroid(long id, int component) const
{
    return m_interfaceCentroids[component].at(id);
}

/*!
 * Gets the centroid of the specified interface.
 *
 * \param id is the id of the interface
 * \param component is the requested component
 * \result The centroid of the specified interface.
 */
std::array<double, 3> MeshGeometricalInfo::getInterfaceCentroid(long id) const
{
    std::size_t rawInterfaceId = m_patch->getInterfaces().find(id).getRawIndex();

    return rawGetInterfaceCentroid(rawInterfaceId);
}

/*!
 * Gets the requested centroid component for the specified interface.
 *
 * \param pos is the raw position of the item
 * \param component is the requested component
 * \result The requested centroid component for the specified interface.
 */
double MeshGeometricalInfo::rawGetInterfaceCentroid(size_t pos, int component) const
{
    return m_interfaceCentroids[component].rawAt(pos);
}

/*!
 * Gets the centroid of the specified interface.
 *
 * \param pos is the raw position of the item
 * \param component is the requested component
 * \result The centroid of the specified interface.
 */
std::array<double, 3> MeshGeometricalInfo::rawGetInterfaceCentroid(size_t pos) const
{
    return {{m_interfaceCentroids[0].rawAt(pos), m_interfaceCentroids[1].rawAt(pos), m_interfaceCentroids[2].rawAt(pos)}};
}

/*!
 * Gets a constant reference to the storage for the interface centroids along
 * the given direction.
 *
 * \param component is the requested component
 * \result A constant reference to the storage for the interface centroids
 * along the given direction.
 */
const bitpit::PiercedStorage<double, long> & MeshGeometricalInfo::getInterfaceCentroids(int component) const
{
    return m_interfaceCentroids[component];
}

/*!
 * Gets a reference to the storage for the interface centroids along the given
 * direction.
 *
 * \param component is the requested component
 * \result A reference to the storage for the interface centroids along the
 * given direction.
 */
bitpit::PiercedStorage<double, long> & MeshGeometricalInfo::getInterfaceCentroids(int component)
{
    return m_interfaceCentroids[component];
}

/*!
 * Gets the requested normal component for the specified interface.
 *
 * \param id is the id of the interface
 * \param component is the requested component
 * \result The requested normal component for the specified interface.
 */
double MeshGeometricalInfo::getInterfaceNormal(long id, int component) const
{
    return m_interfaceNormals[component].at(id);
}

/*!
 * Gets the normal of the specified interface.
 *
 * \param id is the id of the interface
 * \param component is the requested component
 * \result The normal of the specified interface.
 */
std::array<double, 3> MeshGeometricalInfo::getInterfaceNormal(long id) const
{
    std::size_t rawInterfaceId = m_patch->getInterfaces().find(id).getRawIndex();

    return rawGetInterfaceNormal(rawInterfaceId);
}

/*!
 * Gets the requested normal component for the specified interface.
 *
 * \param pos is the raw position of the item
 * \param component is the requested component
 * \result The requested normal component for the specified interface.
 */
double MeshGeometricalInfo::rawGetInterfaceNormal(size_t pos, int component) const
{
    return m_interfaceNormals[component].rawAt(pos);
}

/*!
 * Gets the normal of the specified interface.
 *
 * \param pos is the raw position of the item
 * \param component is the requested component
 * \result The normal of the specified interface.
 */
std::array<double, 3> MeshGeometricalInfo::rawGetInterfaceNormal(size_t pos) const
{
    return {{m_interfaceNormals[0].rawAt(pos), m_interfaceNormals[1].rawAt(pos), m_interfaceNormals[2].rawAt(pos)}};

}

/*!
 * Gets a constant reference to the storage for the interface normal along
 * the given direction.
 *
 * \param component is the requested component
 * \result A constant reference to the storage for the interface normal along
 * the given direction.
 */
const bitpit::PiercedStorage<double, long> & MeshGeometricalInfo::getInterfaceNormals(int component) const
{
    return m_interfaceNormals[component];
}

/*!
 * Gets a reference to the storage for the interface normal along the given
 * direction.
 *
 * \param component is the requested component
 * \result A reference to the storage for the interface normal along the given
 * direction.
 */
bitpit::PiercedStorage<double, long> & MeshGeometricalInfo::getInterfaceNormals(int component)
{
    return m_interfaceNormals[component];
}

/*!
 * Gets the requested tangent component for the specified interface.
 *
 * \param id is the id of the interface
 * \param component is the requested component
 * \result The requested tangent component for the specified interface.
 */
double MeshGeometricalInfo::getInterfaceTangent(long id, int component) const
{
    return m_interfaceTangents[component].at(id);
}

/*!
 * Gets the tangent of the specified interface.
 *
 * \param id is the id of the interface
 * \param component is the requested component
 * \result The requested tangent component for the specified interface.
 */
std::array<double, 3> MeshGeometricalInfo::getInterfaceTangent(long id) const
{
    std::size_t rawInterfaceId = m_patch->getInterfaces().find(id).getRawIndex();

    return rawGetInterfaceTangent(rawInterfaceId);
}

/*!
 * Gets the requested tangent component for the specified interface.
 *
 * \param pos is the raw position of the item
 * \param component is the requested component
 * \result The requested tangent component for the specified interface.
 */
double MeshGeometricalInfo::rawGetInterfaceTangent(size_t pos, int component) const
{
    return m_interfaceTangents[component].rawAt(pos);
}

/*!
 * Gets the tangent of the specified interface.
 *
 * \param pos is the raw position of the item
 * \param component is the requested component
 * \result The tangent of the specified interface.
 */
std::array<double, 3> MeshGeometricalInfo::rawGetInterfaceTangent(size_t pos) const
{
    return {{m_interfaceTangents[0].rawAt(pos), m_interfaceTangents[1].rawAt(pos), m_interfaceTangents[2].rawAt(pos)}};
}

/*!
 * Gets a constant reference to the storage for the interface tangent along
 * the given direction.
 *
 * \param component is the requested component
 * \result A constant reference to the storage for the interface tangents along
 * the given direction.
 */
const bitpit::PiercedStorage<double, long> & MeshGeometricalInfo::getInterfaceTangents(int component) const
{
    return m_interfaceTangents[component];
}

/*!
 * Gets a reference to the storage for the interface tangent along the given
 * direction.
 *
 * \param component is the requested component
 * \result A reference to the storage for the interface tangents along the
 * given direction.
 */
bitpit::PiercedStorage<double, long> & MeshGeometricalInfo::getInterfaceTangents(int component)
{
    return m_interfaceTangents[component];
}
