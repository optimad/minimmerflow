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

#ifndef __MINIMMERFLOW_CONTAINERS_TPP__
#define __MINIMMERFLOW_CONTAINERS_TPP__

#include "storage.hpp"

/*!
 * Constructor
 *
 * \param nStorages is the number of storages
 * \param args are the arguments that will be passed to the constructor of the
 * storages
 */
template<typename storage_t>
template<typename... Args>
BaseStorageCollection<storage_t>::BaseStorageCollection(std::size_t nStorages, Args&&... args)
    : m_storages(nStorages, storage_t(std::forward<Args>(args)...))
{
}

/*!
    Returns a constant reference to the storage of the specified storage.

    \param index is the index of the storage
    \result A constant reference to the storage of the specified storage.
*/
template<typename storage_t>
const typename BaseStorageCollection<storage_t>::storage_type & BaseStorageCollection<storage_t>::operator[](std::size_t index) const
{
    return m_storages[index];
}

/*!
    Returns a reference to the storage of the specified storage.

    \param index is the index of the storage
    \result A reference to the storage of the specified storage.
*/
template<typename storage_t>
typename BaseStorageCollection<storage_t>::storage_type & BaseStorageCollection<storage_t>::operator[](std::size_t index)
{
    return m_storages[index];
}

#if ENABLE_CUDA
/*!
 * Allocate cuda memory.
 */
template<typename storage_t>
void BaseStorageCollection<storage_t>::cuda_allocateDevice()
{
    for (auto &storage : m_storages) {
        storage.cuda_allocateDevice();
    }
}

/*!
 * Free cuda memory.
 */
template<typename storage_t>
void BaseStorageCollection<storage_t>::cuda_freeDevice()
{
    for (auto &storage : m_storages) {
        storage.cuda_freeDevice();
    }
}

/*!
 * Update host data using device data.
 */
template<typename storage_t>
void BaseStorageCollection<storage_t>::cuda_updateHost()
{
    for (auto &storage : m_storages) {
        storage.cuda_updateHost();
    }
}

/*!
 * Update device data using host data.
 */
template<typename storage_t>
void BaseStorageCollection<storage_t>::cuda_updateDevice()
{
    for (auto &storage : m_storages) {
        storage.cuda_updateDevice();
    }
}

/*!
 * Gets a pointer to the device data storage.
 *
 * \result A pointer to the device data storage.
 */
template<typename storage_t>
std::vector<typename BaseStorageCollection<storage_t>::dev_value_type *> BaseStorageCollection<storage_t>::cuda_deviceData()
{
    std::vector<dev_value_type *> deviceData(N_FIELDS);
    for (std::size_t k = 0; k < m_storages.size(); ++k) {
        deviceData[k] = m_storages[k].cuda_deviceData();
    }

    return deviceData;
}

/*!
 * Gets a constant pointer to the device data storage.
 *
 * \result A constant pointer to the device data storage.
 */
template<typename storage_t>
std::vector<const typename BaseStorageCollection<storage_t>::dev_value_type *> BaseStorageCollection<storage_t>::cuda_deviceData() const
{
    std::vector<const dev_value_type *> deviceData(N_FIELDS);
    for (std::size_t k = 0; k < m_storages.size(); ++k) {
        deviceData[k] = m_storages[k].cuda_deviceData();
    }

    return deviceData;
}

/*!
 * Fill the container with the specified value.
 *
 * \param value is the value that will be used to fille the container
 */
template<typename storage_t>
void BaseStorageCollection<storage_t>::cuda_fillDevice(const dev_value_type &value)
{
    for (auto &storage : m_storages) {
        storage.cuda_fillDevice(value);
    }
}
#endif

/*!
 * Constructor
 *
 * \param nStorages is the number of storages
 * \param kernel is the kernel
 */
template<typename value_t, typename dev_value_t, typename id_t>
PiercedStorageCollection<value_t, dev_value_t, id_t>::PiercedStorageCollection(std::size_t nStorages, bitpit::PiercedKernel<id_t> *kernel)
    : BaseStorageCollection<ValuePiercedStorage<value_t, dev_value_t>>(nStorages, 1, kernel)
{
}

/*!
 * Constructor
 *
 * \param nStorages is the number of storages
 */
template<typename value_t, typename dev_value_t>
StorageCollection<value_t, dev_value_t>::StorageCollection(std::size_t nStorages)
    : BaseStorageCollection<ValueStorage<value_t>>(nStorages)
{
}

#endif
