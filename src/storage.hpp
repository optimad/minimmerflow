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

#ifndef __MINIMMERFLOW_STORAGE_HPP__
#define __MINIMMERFLOW_STORAGE_HPP__

#include <constants.hpp>
#include <communications.hpp>

typedef bitpit::PiercedStorage<bool, long> CellStorageBool;
typedef bitpit::PiercedStorage<double, long> CellStorageDouble;

typedef bitpit::PiercedStorage<int, long> InterfaceStorageInt;
typedef bitpit::PiercedStorage<double, long> InterfaceStorageDouble;

#if ENABLE_MPI
typedef PiercedStorageBufferStreamer<double> CellBufferStreamer;
#endif

typedef std::array<double, N_FIELDS> FluxData;

#endif
