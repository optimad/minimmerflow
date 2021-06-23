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

#ifndef __MINIMMERFLOW_CONTAINERS_HPP__
#define __MINIMMERFLOW_CONTAINERS_HPP__

#include <communications.hpp>
#include <utils_cuda.hpp>

#include <bitpit_containers.hpp>

#if ENABLE_OPENACC
#include <openacc.h>
#endif
#include <vector>
#include <array>

template<typename container_t, typename value_t, typename dev_value_t>
class ValueBaseStorage : public container_t
{

public:
    typedef container_t container_type;
    typedef value_t value_type;
    typedef dev_value_t dev_value_type;

#if ENABLE_CUDA
    void cuda_allocateDevice();
    void cuda_freeDevice();

    void cuda_updateHost();
    void cuda_updateDevice();

    dev_value_t * cuda_deviceData();
    const dev_value_t * cuda_deviceData() const;
    virtual std::size_t cuda_deviceDataSize() const = 0;

    void cuda_fillDevice(const dev_value_t &value);
#endif

protected:
    using container_t::container_t;

#if ENABLE_CUDA
    dev_value_t *m_deviceData;
#endif

};

// Avoid implicit instantiation
extern template class ValueBaseStorage<bitpit::PiercedStorage<int, long>, int, int>;
extern template class ValueBaseStorage<bitpit::PiercedStorage<std::size_t, long>, std::size_t, std::size_t>;
extern template class ValueBaseStorage<bitpit::PiercedStorage<double, long>, double, double>;
extern template class ValueBaseStorage<bitpit::PiercedStorage<std::array<double, 3>, long>, std::array<double, 3>, double>;

extern template class ValueBaseStorage<std::vector<int>, int, int>;
extern template class ValueBaseStorage<std::vector<std::size_t>, std::size_t, std::size_t>;
extern template class ValueBaseStorage<std::vector<double>, double, double>;
extern template class ValueBaseStorage<std::vector<std::array<double, 3>>, std::array<double, 3>, double>;

template<typename value_t, typename dev_value_t = value_t>
class ValuePiercedStorage : public ValueBaseStorage<bitpit::PiercedStorage<value_t, long>, value_t, dev_value_t>
{

public:
    using ValueBaseStorage<bitpit::PiercedStorage<value_t, long>, value_t, dev_value_t>::ValueBaseStorage;

#if ENABLE_CUDA
    std::size_t cuda_deviceDataSize() const override;
#endif

};

template<typename value_t>
using ScalarPiercedStorage = ValuePiercedStorage<value_t>;

template<typename value_t>
using VectorPiercedStorage = ValuePiercedStorage<std::array<value_t, 3>, value_t>;

// Avoid implicit instantiation
extern template class ValuePiercedStorage<int, int>;
extern template class ValuePiercedStorage<std::size_t, std::size_t>;
extern template class ValuePiercedStorage<double, double>;
extern template class ValuePiercedStorage<std::array<double, 3>, double>;

#if ENABLE_MPI
template<typename value_t>
using ValuePiercedStorageBufferStreamer = PiercedStorageBufferStreamer<value_t>;
#endif

template<typename value_t, typename dev_value_t = value_t>
class ValueStorage : public ValueBaseStorage<std::vector<value_t>, value_t, dev_value_t>
{

public:
    using ValueBaseStorage<std::vector<value_t>, value_t, dev_value_t>::ValueBaseStorage;

#if ENABLE_CUDA
    void cuda_updateHost(std::size_t count, std::size_t offset = 0);
    using ValueBaseStorage<std::vector<value_t>, value_t, dev_value_t>::cuda_updateHost;

    void cuda_updateDevice(std::size_t count, std::size_t offset = 0);
    using ValueBaseStorage<std::vector<value_t>, value_t, dev_value_t>::cuda_updateDevice;

    std::size_t cuda_deviceDataSize() const override;
#endif

protected:
#if ENABLE_CUDA
    std::size_t cuda_deviceDataSize(std::size_t count) const;
#endif

};

template<typename value_t>
using ScalarStorage = ValueStorage<value_t>;

template<typename value_t>
using VectorStorage = ValueStorage<std::array<value_t, 3>, value_t>;

// Avoid implicit instantiation
extern template class ValueStorage<int, int>;
extern template class ValueStorage<std::size_t, std::size_t>;
extern template class ValueStorage<double, double>;
extern template class ValueStorage<std::array<double, 3>, double>;

#endif
