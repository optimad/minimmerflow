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
    : m_storages(nStorages, storage_t(std::forward<Args>(args)...)),
      m_dataCollection(nStorages)
{
    for (std::size_t i = 0; i < m_storages.size(); ++i) {
        m_dataCollection[i] = m_storages[i].data();
    }
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

/*!
    Gets a pointer to the data collection.

    \result A pointer to the device data collection.
*/
template<typename storage_t>
BaseStorageCollection<storage_t>::value_type ** BaseStorageCollection<storage_t>::collectionData()
{
    return m_dataCollection.data();
}

/*!
    Gets a constant pointer to the device data collection.

    \result A constant pointer to the device data collection.
*/
template<typename storage_t>
const BaseStorageCollection<storage_t>::value_type * const * BaseStorageCollection<storage_t>::collectionData() const
{
    return m_dataCollection.data();
}

/*!
 * Constructor
 *
 * \param nStorages is the number of storages
 */
template<typename value_t, typename dev_value_t, typename id_t>
PiercedStorageCollection<value_t, dev_value_t, id_t>::PiercedStorageCollection(std::size_t nStorages)
    : BaseStorageCollection<ValuePiercedStorage<value_t, dev_value_t>>(nStorages, 1)
{
}

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
    : BaseStorageCollection<ValueStorage<value_t, dev_value_t>>(nStorages)
{
}

#endif
