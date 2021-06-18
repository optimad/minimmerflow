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
    m_cellRawIds.cuda_allocate();
    m_internalCellRawIds.cuda_allocate();

    m_cellVolumes.cuda_allocate();
    m_cellSizes.cuda_allocate();
    m_cellCentroids.cuda_allocate();

    m_interfaceRawIds.cuda_allocate();
    m_interfaceOwnerRawIds.cuda_allocate();
    m_interfaceNeighRawIds.cuda_allocate();

    m_interfaceAreas.cuda_allocate();
    m_interfaceCentroids.cuda_allocate();
    m_interfaceNormals.cuda_allocate();
    m_interfaceTangents.cuda_allocate();

    // Copy data to the device
    m_cellRawIds.cuda_updateDevice();
    m_internalCellRawIds.cuda_updateDevice();

    m_cellVolumes.cuda_updateDevice();
    m_cellSizes.cuda_updateDevice();
    m_cellCentroids.cuda_updateDevice();

    m_interfaceRawIds.cuda_updateDevice();
    m_interfaceOwnerRawIds.cuda_updateDevice();
    m_interfaceNeighRawIds.cuda_updateDevice();

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
    m_cellRawIds.cuda_free();
    m_internalCellRawIds.cuda_free();

    m_cellVolumes.cuda_free();
    m_cellSizes.cuda_free();
    m_cellCentroids.cuda_free();

    m_interfaceRawIds.cuda_free();
    m_interfaceOwnerRawIds.cuda_free();
    m_interfaceNeighRawIds.cuda_free();

    m_interfaceAreas.cuda_free();
    m_interfaceCentroids.cuda_free();
    m_interfaceNormals.cuda_free();
    m_interfaceTangents.cuda_free();
}

/*!
 * Gets a constant pointer to the cell raw id CUDA data storage.
 *
 * \result A constant pointer to the cell raw id CUDA data storage.
 */
const std::size_t * MeshGeometricalInfo::cuda_getCellRawIdDevData() const
{
    return m_cellRawIds.cuda_devData();
}

/*!
 * Gets a constant pointer to the internal cell raw id CUDA data storage.
 *
 * \result A constant pointer to the internal cell raw id CUDA data storage.
 */
const std::size_t * MeshGeometricalInfo::cuda_getInternalCellRawIdDevData() const
{
    return m_internalCellRawIds.cuda_devData();
}

/*!
 * Gets a pointer to the cell volume CUDA data storage.
 *
 * \result A pointer to the cell volume CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getCellVolumeDevData()
{
    return m_cellVolumes.cuda_devData();
}

/*!
 * Gets a constant pointer to the cell volume CUDA data storage.
 *
 * \result A constant pointer to the cell volume CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getCellVolumeDevData() const
{
    return m_cellVolumes.cuda_devData();
}

/*!
 * Gets a pointer to the cell size CUDA data storage.
 *
 * \result A pointer to the cell size CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getCellSizeDevData()
{
    return m_cellSizes.cuda_devData();
}

/*!
 * Gets a constant pointer to the cell size CUDA data storage.
 *
 * \result A constant pointer to the cell size CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getCellSizeDevData() const
{
    return m_cellSizes.cuda_devData();
}

/*!
 * Gets a pointer to the cell centroid CUDA data storage.
 *
 * \result A pointer to the cell centroid CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getCellCentroidDevData()
{
    return m_cellCentroids.cuda_devData();
}

/*!
 * Gets a constant pointer to the cell centroid CUDA data storage.
 *
 * \result A constant pointer to the cell centroid CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getCellCentroidDevData() const
{
    return m_cellCentroids.cuda_devData();
}

/*!
 * Gets a constant pointer to the interface raw id CUDA data storage.
 *
 * \result A constant pointer to the interface raw id CUDA data storage.
 */
const std::size_t * MeshGeometricalInfo::cuda_getInterfaceRawIdDevData() const
{
    return m_interfaceRawIds.cuda_devData();
}

/*!
 * Gets a constant pointer to the interface owner raw id CUDA data storage.
 *
 * \result A constant pointer to the interface owner raw id CUDA data storage.
 */
const std::size_t * MeshGeometricalInfo::cuda_getInterfaceOwnerRawIdDevData() const
{
    return m_interfaceOwnerRawIds.cuda_devData();
}

/*!
 * Gets a constant pointer to the interface neigh raw id CUDA data storage.
 *
 * \result A constant pointer to the interface neigh raw id CUDA data storage.
 */
const std::size_t * MeshGeometricalInfo::cuda_getInterfaceNeighRawIdDevData() const
{
    return m_interfaceNeighRawIds.cuda_devData();
}

/*!
 * Gets a pointer to the interface area CUDA data storage.
 *
 * \result A pointer to the interface area CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getInterfaceAreaDevData()
{
    return m_interfaceAreas.cuda_devData();
}

/*!
 * Gets a constant pointer to the interface area CUDA data storage.
 *
 * \result A constant pointer to the interface area CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getInterfaceAreaDevData() const
{
    return m_interfaceAreas.cuda_devData();
}

/*!
 * Gets a pointer to the interface centroid CUDA data storage.
 *
 * \result A pointer to the interface centroid CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getInterfaceCentroidDevData()
{
    return m_interfaceCentroids.cuda_devData();
}

/*!
 * Gets a constant pointer to the interface centroid CUDA data storage.
 *
 * \result A constant pointer to the interface centroid CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getInterfaceCentroidDevData() const
{
    return m_interfaceCentroids.cuda_devData();
}

/*!
 * Gets a pointer to the interface normal CUDA data storage.
 *
 * \result A pointer to the interface normal CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getInterfaceNormalDevData()
{
    return m_interfaceNormals.cuda_devData();
}

/*!
 * Gets a constant pointer to the interface normal CUDA data storage.
 *
 * \result A constant pointer to the interface normal CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getInterfaceNormalDevData() const
{
    return m_interfaceNormals.cuda_devData();
}

/*!
 * Gets a pointer to the interface tangent CUDA data storage.
 *
 * \result A pointer to the interface tangent CUDA data storage.
 */
double * MeshGeometricalInfo::cuda_getInterfaceTangentDevData()
{
    return m_interfaceTangents.cuda_devData();
}

/*!
 * Gets a constant pointer to the interface tangent CUDA data storage.
 *
 * \result A constant pointer to the interface tangent CUDA data storage.
 */
const double * MeshGeometricalInfo::cuda_getInterfaceTangentDevData() const
{
    return m_interfaceTangents.cuda_devData();
}
