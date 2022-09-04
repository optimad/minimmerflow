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

#include <bitpit_discretization.hpp>

#include <array>
#include <vector>

class ReconstructionCalculator {

public:
    ReconstructionCalculator(const MeshGeometricalInfo &meshInfo, int degree);

    void update(const CellStorageDouble &cellConservatives);

    void evalCellValues(std::size_t cellRawId, const std::array<double, 3> &point, double *values) const;

private:
    const MeshGeometricalInfo &m_meshInfo;
    int m_order;

    bitpit::PiercedStorage<std::vector<long>, long> m_cellSupports;
    bitpit::PiercedStorage<bitpit::ReconstructionKernel, long> m_cellKernels;
    bitpit::PiercedStorage<bitpit::ReconstructionPolynomial, long> m_cellPolynomials;

    void evalCellSupport(std::size_t cellRawId, std::vector<long> *support) const;

    void evalCellKernel(std::size_t cellRawId, std::size_t supportSize, const long *support,
                        bitpit::ReconstructionKernel *kernel) const;

    void initializeCellPolynomial(std::size_t cellRawId, bitpit::ReconstructionPolynomial *polynomial) const;
    void updateCellPolynomial(std::size_t cellRawId, const CellStorageDouble &conservativeFields,
                              bitpit::ReconstructionPolynomial *polynomial) const;

    void evalPolynomial(const bitpit::ReconstructionPolynomial &polynomial,
                        const std::array<double, 3> &point, double *values) const;

};

#endif
