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

/*!
 * Initialize CUDA operations.
 */
void MeshGeometricalInfo::cuda_initialize()
{
    // Allocate CUDA memory
    m_cellVolumes.cuda_allocateDevice();
    m_cellSizes.cuda_allocateDevice();
    m_cellCentroids.cuda_allocateDevice();

    m_interfaceAreas.cuda_allocateDevice();
    m_interfaceCentroids.cuda_allocateDevice();
    m_interfaceNormals.cuda_allocateDevice();
    m_interfaceTangents.cuda_allocateDevice();

    // Copy data to the device
    m_cellVolumes.cuda_updateDevice(m_cellVolumes.cuda_deviceDataSize(), 0);
    m_cellSizes.cuda_updateDevice(m_cellVolumes.cuda_deviceDataSize(), 0);
    m_cellCentroids.cuda_updateDevice();

    m_interfaceAreas.cuda_updateDevice();
    m_interfaceCentroids.cuda_updateDevice();
    m_interfaceNormals.cuda_updateDevice();
    m_interfaceTangents.cuda_updateDevice();
}

/*!
 * Finalize CUDA operations.
 */
void MeshGeometricalInfo::cuda_finalize()
{
    // Deallocate CUDA memory
    m_cellVolumes.cuda_freeDevice();
    m_cellSizes.cuda_freeDevice();
    m_cellCentroids.cuda_freeDevice();

    m_interfaceAreas.cuda_freeDevice();
    m_interfaceCentroids.cuda_freeDevice();
    m_interfaceNormals.cuda_freeDevice();
    m_interfaceTangents.cuda_freeDevice();
}

/*!
 * Resize GPU allocated memory
 */
void MeshGeometricalInfo::cuda_resize()
{
    // Resize GPU arrays
    m_cellVolumes.cuda_resize(m_volumePatch->getCellCount());
    m_cellSizes.cuda_resize(m_volumePatch->getCellCount());
    m_cellCentroids.cuda_resize(m_volumePatch->getCellCount());

    m_interfaceAreas.cuda_resize(m_volumePatch->getInterfaceCount());
    m_interfaceCentroids.cuda_resize(m_volumePatch->getInterfaceCount());
    m_interfaceNormals.cuda_resize(m_volumePatch->getInterfaceCount());
    m_interfaceTangents.cuda_resize(m_volumePatch->getInterfaceCount());
}

/*!
 * Gets a pointer to the cell volume CUDA data storage.
 *
 * \result A pointer to the cell volume CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getCellVolumeDevData()
{
    return m_cellVolumes.cuda_deviceData();
}

/*!
 * Gets a constant pointer to the cell volume CUDA data storage.
 *
 * \result A constant pointer to the cell volume CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getCellVolumeDevData() const
{
    return m_cellVolumes.cuda_deviceData();
}

/*!
 * Gets a pointer to the cell size CUDA data storage.
 *
 * \result A pointer to the cell size CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getCellSizeDevData()
{
    return m_cellSizes.cuda_deviceData();
}

/*!
 * Gets a constant pointer to the cell size CUDA data storage.
 *
 * \result A constant pointer to the cell size CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getCellSizeDevData() const
{
    return m_cellSizes.cuda_deviceData();
}

/*!
 * Gets a pointer to the cell centroid CUDA data storage.
 *
 * \result A pointer to the cell centroid CUDA data storage.
 */
double ** MeshGeometricalInfo::cuda_getCellCentroidDevData()
{
    return m_cellCentroids.cuda_deviceCollectionData();
}

/*!
 * Gets a constant pointer to the cell centroid CUDA data storage.
 *
 * \result A constant pointer to the cell centroid CUDA data storage.
 */
const double * const * MeshGeometricalInfo::cuda_getCellCentroidDevData() const
{
    return m_cellCentroids.cuda_deviceCollectionData();
}

/*!
 * Gets a pointer to the interface area CUDA data storage.
 *
 * \result A pointer to the interface area CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getInterfaceAreaDevData()
{
    return m_interfaceAreas.cuda_deviceData();
}

/*!
 * Gets a constant pointer to the interface area CUDA data storage.
 *
 * \result A constant pointer to the interface area CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getInterfaceAreaDevData() const
{
    return m_interfaceAreas.cuda_deviceData();
}

/*!
 * Gets a pointer to the interface centroid CUDA data storage.
 *
 * \result A pointer to the interface centroid CUDA data storage.
 */
double ** MeshGeometricalInfo::cuda_getInterfaceCentroidDevData()
{
    return m_interfaceCentroids.cuda_deviceCollectionData();
}

/*!
 * Gets a constant pointer to the interface centroid CUDA data storage.
 *
 * \result A constant pointer to the interface centroid CUDA data storage.
 */
const double * const * MeshGeometricalInfo::cuda_getInterfaceCentroidDevData() const
{
    return m_interfaceCentroids.cuda_deviceCollectionData();
}

/*!
 * Gets a pointer to the interface normal CUDA data storage.
 *
 * \result A pointer to the interface normal CUDA data storage.
 */
double ** MeshGeometricalInfo::cuda_getInterfaceNormalDevData()
{
    return m_interfaceNormals.cuda_deviceCollectionData();
}

/*!
 * Gets a constant pointer to the interface normal CUDA data storage.
 *
 * \result A constant pointer to the interface normal CUDA data storage.
 */
const double * const * MeshGeometricalInfo::cuda_getInterfaceNormalDevData() const
{
    return m_interfaceNormals.cuda_deviceCollectionData();
}

/*!
 * Gets a pointer to the interface tangent CUDA data storage.
 *
 * \result A pointer to the interface tangent CUDA data storage.
 */
double ** MeshGeometricalInfo::cuda_getInterfaceTangentDevData()
{
    return m_interfaceTangents.cuda_deviceCollectionData();
}

/*!
 * Gets a constant pointer to the interface tangent CUDA data storage.
 *
 * \result A constant pointer to the interface tangent CUDA data storage.
 */
const double * const * MeshGeometricalInfo::cuda_getInterfaceTangentDevData() const
{
    return m_interfaceTangents.cuda_deviceCollectionData();
}
