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

#ifndef __MINIMMERFLOW_COMPUTATION_INFO_HPP__
#define __MINIMMERFLOW_COMPUTATION_INFO_HPP__

#include "containers.hpp"
#include "mesh_info.hpp"

#include <bitpit_patchkernel.hpp>

class ComputationInfo : public MeshGeometricalInfo {

public:
    ComputationInfo(bitpit::VolumeKernel *patch);

    const ScalarPiercedStorage<int> & getCellSolveMethods() const;
    ScalarPiercedStorage<int> & getCellSolveMethods();

    const ScalarStorage<std::size_t> & getSolvedCellRawIds() const;
    ScalarStorage<std::size_t> & getSolvedCellRawIds();

    const ScalarPiercedStorage<int> & getInterfaceSolveMethods() const;

    const ScalarStorage<std::size_t> & getSolvedUniformInterfaceRawIds() const;
    const ScalarStorage<std::size_t> & getSolvedUniformInterfaceOwnerRawIds() const;
    const ScalarStorage<std::size_t> & getSolvedUniformInterfaceNeighRawIds() const;

    const ScalarStorage<std::size_t> & getSolvedBoundaryInterfaceRawIds() const;
    ScalarStorage<std::size_t> & getSolvedBoundaryInterfaceRawIds();
    const ScalarStorage<std::size_t> & getSolvedBoundaryInterfaceFluidRawIds() const;
    const ScalarStorage<int> & getSolvedBoundaryInterfaceSigns() const;

#if ENABLE_CUDA
    void cuda_initialize() override;
    void cuda_finalize() override;
    void cuda_resize() override;
#endif

    void clearScalarStorages();
    void postMeshAdaptation();

protected:
    ScalarPiercedStorage<int> m_cellSolveMethods;

    ScalarStorage<std::size_t> m_solvedCellRawIds;

    ScalarPiercedStorage<int> m_interfaceSolveMethods;

    ScalarStorage<std::size_t> m_solvedUniformInterfaceRawIds;
    ScalarStorage<std::size_t> m_solvedUniformInterfaceOwnerRawIds;
    ScalarStorage<std::size_t> m_solvedUniformInterfaceNeighRawIds;

    ScalarStorage<std::size_t> m_solvedBoundaryInterfaceRawIds;
    ScalarStorage<std::size_t> m_solvedBoundaryInterfaceFluidRawIds;
    ScalarStorage<int> m_solvedBoundaryInterfaceSigns;

    void _init() override;
    void _reset() override;
    void _extract() override;

private:
    bool isInterfaceBoundary(const bitpit::Interface &interface) const;

};

#endif
