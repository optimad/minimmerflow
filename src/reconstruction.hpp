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

#ifndef __MINIMMERFLOW_RECONSTRUCTION_HPP__
#define __MINIMMERFLOW_RECONSTRUCTION_HPP__

#include "constants.hpp"
#include "mesh_info.hpp"
#include "problem.hpp"
#include "storage.hpp"

#include <bitpit_voloctree.hpp>

#include <array>

namespace reconstruction {

void initialize();

void computePolynomials(problem::ProblemType problemType, const MeshGeometricalInfo &meshInfo, const ScalarPiercedStorage<bool> &cellSolved,
                        const ScalarPiercedStorage<double> &conservativeFields, const ScalarPiercedStorage<int> &interfaceBCs);

void eval(std::size_t cellRawId, const MeshGeometricalInfo &meshInfo, int order, const std::array<double, 3> &point, const double *means, double *values);

void eval_1(std::size_t cellRawId, const MeshGeometricalInfo &meshInfo, const std::array<double, 3> &point, const double *means, double *values);

}

#endif
