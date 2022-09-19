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
#include "compiler.hpp"
#include "computation_info.hpp"
#include "problem.hpp"
#include "storage.hpp"

#include <bitpit_discretization.hpp>

#include <vector>

class ReconstructionCalculator
{

public:
    ReconstructionCalculator(const ComputationInfo &computationInfo, int degree);

    int getOrder() const;
    int getDimension() const;

    const ScalarPiercedStorageCollection<double> & getCellPolynomials() const;
    ScalarPiercedStorageCollection<double> & getCellPolynomials();
#if ENABLE_CUDA
    double ** cuda_getCellPolynomialDevData();
    const double * const * cuda_getCellPolynomialDevData() const;
#endif

    void update(const ScalarPiercedStorageCollection<double> &cellConservatives);

#if ENABLE_CUDA
    void cuda_initialize();
    void cuda_finalize();
#endif

private:
    const ComputationInfo &m_computationInfo;

    int m_order;
    int m_dimension;
    int m_nBasis;

    ScalarPiercedStorage<std::size_t> m_cellSupportSizes;
    ScalarPiercedStorage<std::size_t> m_cellSupportOffsets;
    ScalarStorage<long> m_cellSupportRawIds;

    ScalarStorage<double> m_cellKernelWeights;

    ScalarPiercedStorageCollection<double> m_cellPolynomials;

#if ENABLE_CUDA
    void cuda_updateCellPolynomials(const ScalarPiercedStorageCollection<double> &cellConservatives);
#else
    void updateCellPolynomials(const ScalarPiercedStorageCollection<double> &cellConservatives);
#endif

    void evalCellSupport(std::size_t cellRawId, std::size_t *supportSize, long *supportIds = nullptr) const;

    void evalCellKernel(std::size_t cellRawId, std::size_t supportSize, const long *support,
                        bitpit::ReconstructionKernel *kernel) const;

};

#endif
