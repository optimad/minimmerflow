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

#include "containers.hpp"

#if ENABLE_CUDA == 0
// Explicit instantiation
template class ValueBaseStorage<bitpit::PiercedStorage<int, long>, int, int>;
template class ValueBaseStorage<bitpit::PiercedStorage<long, long>, long, long>;
template class ValueBaseStorage<bitpit::PiercedStorage<std::size_t, long>, std::size_t, std::size_t>;
template class ValueBaseStorage<bitpit::PiercedStorage<double, long>, double, double>;
template class ValueBaseStorage<bitpit::PiercedStorage<std::array<double, 3>, long>, std::array<double, 3>, double>;

template class ValueBaseStorage<std::vector<int>, int, int>;
template class ValueBaseStorage<std::vector<long>, long, long>;
template class ValueBaseStorage<std::vector<std::size_t>, std::size_t, std::size_t>;
template class ValueBaseStorage<std::vector<double>, double, double>;
template class ValueBaseStorage<std::vector<std::array<double, 3>>, std::array<double, 3>, double>;

template class ValuePiercedStorage<int>;
template class ValuePiercedStorage<long>;
template class ValuePiercedStorage<std::size_t>;
template class ValuePiercedStorage<double>;
template class ValuePiercedStorage<std::array<double, 3>, double>;

template class ValueStorage<int>;
template class ValueStorage<long>;
template class ValueStorage<std::size_t>;
template class ValueStorage<double>;
template class ValueStorage<std::array<double, 3>, double>;

template class BaseStorageCollection<ValuePiercedStorage<int, int>>;
template class BaseStorageCollection<ValuePiercedStorage<long, long>>;
template class BaseStorageCollection<ValuePiercedStorage<std::size_t, std::size_t>>;
template class BaseStorageCollection<ValuePiercedStorage<double, double>>;
template class BaseStorageCollection<ValuePiercedStorage<std::array<double, 3>, double>>;

template class BaseStorageCollection<ValueStorage<int, int>>;
template class BaseStorageCollection<ValueStorage<long, long>>;
template class BaseStorageCollection<ValueStorage<std::size_t, std::size_t>>;
template class BaseStorageCollection<ValueStorage<double, double>>;
template class BaseStorageCollection<ValueStorage<std::array<double, 3>, double>>;

template class PiercedStorageCollection<int, int>;
template class PiercedStorageCollection<long, long>;
template class PiercedStorageCollection<std::size_t, std::size_t>;
template class PiercedStorageCollection<double, double>;
template class PiercedStorageCollection<std::array<double, 3>, double>;

template class StorageCollection<int>;
template class StorageCollection<long>;
template class StorageCollection<std::size_t>;
template class StorageCollection<double>;
template class StorageCollection<std::array<double, 3>, double>;
#endif
