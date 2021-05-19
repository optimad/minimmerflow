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

#if ENABLE_MPI

#ifndef __MINIMMERFLOW_COMMUNICATIONS_TPP__
#define __MINIMMERFLOW_COMMUNICATIONS_TPP__

#include <bitpit_IO.hpp>

/*!
    \class BaseListBufferStreamer

    \brief The BaseListBufferStreamer class allows to define streamers
    to stream list data from / to the buffer of a ListCommunicator.
*/

/*!
    Creates a new streamer

    \param container is the container that holds the data that will be
    exchanged
*/
template<typename container_t>
BaseListBufferStreamer<container_t>::BaseListBufferStreamer(container_t *container)
    : ExchangeBufferStreamer(sizeof(typename container_t::value_type)),
      m_container(container)
{
}

/*!
    Creates a new streamer

    \param container is the container that holds the data that will be
    exchanged
    \param itemSize is the size, expressed in bytes, of the single item that
    will be exchanged
*/
template<typename container_t>
BaseListBufferStreamer<container_t>::BaseListBufferStreamer(container_t *container, const size_t &itemSize)
    : ExchangeBufferStreamer(itemSize),
      m_container(container)
{
}

/*!
    Gets a reference to the container that holds the data that will be
    streamed.

    \result A reference to the container that holds the data that will be
    streamed.
*/
template<typename container_t>
container_t & BaseListBufferStreamer<container_t>::getContainer()
{
    return *m_container;
}

/*!
    \class ListBufferStreamer

    \brief The ListBufferStreamer class allows to stream list data from / to
    the buffer of a ListCommunicator.
*/

/*!
    Read the dataset from the buffer.

    \param rank is the rank of the process who sent the data
    \param buffer is the buffer where the data will be read from
    \param list is the list of ids that will be read
*/
template<typename container_t>
void ListBufferStreamer<container_t>::read(int const &rank, bitpit::RecvBuffer &buffer,
                                           const std::vector<long> &list)
{
    BITPIT_UNUSED(rank);

    container_t &container = this->getContainer();
    for (const long k : list) {
        buffer >> container[k];
    }
}

/*!
    Write the dataset to the buffer.

    \param rank is the rank of the process who will receive the data
    \param buffer is the buffer where the data will be written to
    \param list is the list of ids that will be written
*/
template<typename container_t>
void ListBufferStreamer<container_t>::write(const int &rank, bitpit::SendBuffer &buffer,
                                            const std::vector<long> &list)
{
    BITPIT_UNUSED(rank);

    container_t &container = this->getContainer();
    for (const long k : list) {
        buffer << container[k];
    }
}

/*!
 * \class PiercedStorageBufferStreamer
 *
 * \brief The PiercedStorageBufferStreamer class allows to write and read the
 * data contained in a PiercedStorage to/from the buffer of a DataCommunicator.
 */

/*!
    Creates a new streamer

    \param container is the container that holds the data that will be
    exchanged
*/
template<typename value_t>
PiercedStorageBufferStreamer<value_t>::PiercedStorageBufferStreamer(bitpit::PiercedStorage<value_t, long> *container)
    : ListBufferStreamer<bitpit::PiercedStorage<value_t, long>>(container, container->getFieldCount() * sizeof(typename bitpit::PiercedStorage<value_t, long>::value_type)){
}

/*!
    Creates a new streamer

    \param container is the container that holds the data that will be
    exchanged
    \param itemSize is the size, expressed in bytes, of the single item that
    will be exchanged
*/
template<typename value_t>
PiercedStorageBufferStreamer<value_t>::PiercedStorageBufferStreamer(bitpit::PiercedStorage<value_t, long> *container, const size_t &itemSize)
    : ListBufferStreamer<bitpit::PiercedStorage<value_t, long>>(container, container->getFieldCount() * itemSize)
{
}

/*!
    Read the dataset from the buffer.

    \param rank is the rank of the process who sent the data
    \param buffer is the buffer where the data will be read from
    \param list is the list of ids that will be read
*/
template<typename value_t>
void PiercedStorageBufferStreamer<value_t>::read(int const &rank, bitpit::RecvBuffer &buffer,
                                           const std::vector<long> &list)
{
    BITPIT_UNUSED(rank);

    bitpit::PiercedStorage<value_t, long> &container = this->getContainer();
    std::size_t nFields = container.getFieldCount();
    for (const long k : list) {
        for (std::size_t i = 0; i < nFields; ++i) {
            buffer >> container.at(k, i);
        }
    }
}

/*!
    Write the dataset to the buffer.

    \param rank is the rank of the process who will receive the data
    \param buffer is the buffer where the data will be written to
    \param list is the list of ids that will be written
*/
template<typename value_t>
void PiercedStorageBufferStreamer<value_t>::write(const int &rank, bitpit::SendBuffer &buffer,
                                            const std::vector<long> &list)
{
    BITPIT_UNUSED(rank);

    bitpit::PiercedStorage<value_t, long> &container = this->getContainer();
    std::size_t nFields = container.getFieldCount();
    for (const long k : list) {
        for (std::size_t i = 0; i < nFields; ++i) {
            buffer << container.at(k, i);
        }
    }
}

#endif

#endif
