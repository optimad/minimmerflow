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

#include <bitpit_containers.hpp>

#include <vector>
#include <array>

#if ENABLE_CUDA
template<typename container_t, typename value_t, typename dev_value_t>
class ValueBaseStorage : public container_t
{

public:
    void cuda_allocate();
    void cuda_free();

    void cuda_updateHost();
    void cuda_updateDevice();

    dev_value_t * cuda_devData();
    const dev_value_t * cuda_devData() const;

    virtual std::size_t cuda_devSize() const = 0;

    void cuda_devFill(const dev_value_t &value);

protected:
    using container_t::container_t;
    
    dev_value_t *m_devData;

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
#endif

#if ENABLE_CUDA
template<typename value_t, typename dev_value_t = value_t>
class ValuePiercedStorage : public ValueBaseStorage<bitpit::PiercedStorage<value_t, long>, value_t, dev_value_t>
{

public:
    std::size_t cuda_devSize() const override;
    
    using ValueBaseStorage<bitpit::PiercedStorage<value_t, long>, value_t, dev_value_t>::ValueBaseStorage;

};
#else
template<typename value_t, typename dev_value_t = value_t>
using ValuePiercedStorage = bitpit::PiercedStorage<value_t, long>;
#endif

template<typename value_t>
using ScalarPiercedStorage = ValuePiercedStorage<value_t>;

template<typename value_t>
using VectorPiercedStorage = ValuePiercedStorage<std::array<value_t, 3>, value_t>;

#if ENABLE_CUDA
// Avoid implicit instantiation
extern template class ValuePiercedStorage<int, int>;
extern template class ValuePiercedStorage<std::size_t, std::size_t>;
extern template class ValuePiercedStorage<double, double>;
extern template class ValuePiercedStorage<std::array<double, 3>, double>;
#endif

#if ENABLE_MPI
template<typename value_t>
using ValuePiercedStorageBufferStreamer = PiercedStorageBufferStreamer<value_t>;
#endif

#if ENABLE_CUDA
template<typename value_t, typename dev_value_t = value_t>
class ValueStorage : public ValueBaseStorage<std::vector<value_t>, value_t, dev_value_t>
{

public:
    using ValueBaseStorage<std::vector<value_t>, value_t, dev_value_t>::ValueBaseStorage;
    
    std::size_t cuda_devSize() const override;

};
#else
template<typename value_t, typename dev_value_t = value_t>
using ValueStorage = std::vector<value_t>;
#endif

template<typename value_t>
using ScalarStorage = ValueStorage<value_t>;

template<typename value_t>
using VectorStorage = ValueStorage<std::array<value_t, 3>, value_t>;

#if ENABLE_CUDA
// Avoid implicit instantiation
extern template class ValueStorage<int, int>;
extern template class ValueStorage<std::size_t, std::size_t>;
extern template class ValueStorage<double, double>;
extern template class ValueStorage<std::array<double, 3>, double>;
#endif

#endif
