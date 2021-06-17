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
template<typename value_t>
class ValuePiercedStorage : public bitpit::PiercedStorage<value_t, long>
{

public:
    using bitpit::PiercedStorage<value_t, long>::PiercedStorage;

    void cuda_allocate();
    void cuda_free();

    void cuda_updateHost();
    void cuda_updateDevice();

    double * cuda_devData();
    const double * cuda_devData() const;

    void cuda_devFill(const value_t &value);

private:
    double *m_devData;

};
#else
template<typename value_t>
using ValuePiercedStorage = bitpit::PiercedStorage<value_t, long>;
#endif

template<typename value_t>
using ScalarPiercedStorage = ValuePiercedStorage<value_t>;

template<typename value_t>
using VectorPiercedStorage = ValuePiercedStorage<std::array<value_t, 3>>;

#if ENABLE_CUDA
template<typename value_t>
class ValueStorage : public std::vector<value_t>
{

public:
    using std::vector<value_t>::vector;

    void cuda_allocate();
    void cuda_free();

    void cuda_updateHost();
    void cuda_updateDevice();

    double * cuda_devData();
    const double * cuda_devData() const;

    void cuda_devFill(const value_t &value);

private:
    double *m_devData;

};
#else
template<typename value_t>
using ValueStorage = std::vector<value_t>;
#endif

template<typename value_t>
using ScalarStorage = ValueStorage<value_t>;

template<typename value_t>
using VectorStorage = ValueStorage<std::array<value_t, 3>>;

#if ENABLE_MPI
template<typename value_t>
using ValuePiercedStorageBufferStreamer = PiercedStorageBufferStreamer<value_t>;
#endif

#if ENABLE_CUDA
// Include template implementations
#include <containers.tcu>
#endif

#endif
