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

#ifndef __MINIMMERFLOW_MESH_INFO_HPP__
#define __MINIMMERFLOW_MESH_INFO_HPP__

#include "containers.hpp"

#include <bitpit_patchkernel.hpp>

class MeshGeometricalInfo : public bitpit::PatchInfo {

public:
    MeshGeometricalInfo(bitpit::VolumeKernel *patch = nullptr);

    void setPatch(bitpit::VolumeKernel const *patch);
    bitpit::VolumeKernel const & getPatch() const;

    int getDimension() const;

    const ScalarStorage<std::size_t> & getCellRawIds() const;
#if ENABLE_CUDA
    const std::size_t * cuda_getCellRawIdDevData() const;
#endif
    const ScalarStorage<std::size_t> & getInternalCellRawIds() const;
#if ENABLE_CUDA
    const std::size_t * cuda_getInternalCellRawIdDevData() const;
#endif

    double getCellVolume(long id) const;
    double rawGetCellVolume(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getCellVolumes() const;
    bitpit::PiercedStorage<double, long> & getCellVolumes();
#if ENABLE_CUDA
    double * cuda_getCellVolumeDevData();
    const double * cuda_getCellVolumeDevData() const;
#endif

    double getCellSize(long id) const;
    double rawGetCellSize(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getCellSizes() const;
    bitpit::PiercedStorage<double, long> & getCellSizes();
#if ENABLE_CUDA
    double * cuda_getCellSizeDevData();
    const double * cuda_getCellSizeDevData() const;
#endif

    const std::array<double, 3> & getCellCentroid(long id) const;
    const std::array<double, 3> & rawGetCellCentroid(size_t pos) const;
    const bitpit::PiercedStorage<std::array<double, 3>, long> & getCellCentroids() const;
    bitpit::PiercedStorage<std::array<double, 3>, long> & getCellCentroids();
#if ENABLE_CUDA
    double * cuda_getCellCentroidDevData();
    const double * cuda_getCellCentroidDevData() const;
#endif

    const ScalarStorage<std::size_t> & getInterfaceRawIds() const;
#if ENABLE_CUDA
    const std::size_t * cuda_getInterfaceRawIdDevData() const;
#endif
    const ScalarStorage<std::size_t> & getInterfaceOwnerRawIds() const;
#if ENABLE_CUDA
    const std::size_t * cuda_getInterfaceOwnerRawIdDevData() const;
#endif
    const ScalarStorage<std::size_t> & getInterfaceNeighRawIds() const;
#if ENABLE_CUDA
    const std::size_t * cuda_getInterfaceNeighRawIdDevData() const;
#endif

    double getInterfaceArea(long id) const;
    double rawGetInterfaceArea(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getInterfaceAreas() const;
    bitpit::PiercedStorage<double, long> & getInterfaceAreas();
#if ENABLE_CUDA
    double * cuda_getInterfaceAreaDevData();
    const double * cuda_getInterfaceAreaDevData() const;
#endif

    const std::array<double, 3> & getInterfaceCentroid(long id) const;
    const std::array<double, 3> & rawGetInterfaceCentroid(size_t pos) const;
    const bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceCentroids() const;
    bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceCentroids();
#if ENABLE_CUDA
    double * cuda_getInterfaceCentroidDevData();
    const double * cuda_getInterfaceCentroidDevData() const;
#endif

    const std::array<double, 3> & getInterfaceNormal(long id) const;
    const std::array<double, 3> & rawGetInterfaceNormal(size_t pos) const;
    const bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceNormals() const;
    bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceNormals();
#if ENABLE_CUDA
    double * cuda_getInterfaceNormalDevData();
    const double * cuda_getInterfaceNormalDevData() const;
#endif

    const std::array<double, 3> & getInterfaceTangent(long id) const;
    const std::array<double, 3> & rawGetInterfaceTangent(size_t pos) const;
    const bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceTangents() const;
    bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceTangents();
#if ENABLE_CUDA
    double * cuda_getInterfaceTangentDevData();
    const double * cuda_getInterfaceTangentDevData() const;
#endif

#if ENABLE_CUDA
    void cuda_initialize();
    void cuda_finalize();
#endif

protected:
    const bitpit::VolumeKernel *m_volumePatch;

    ScalarStorage<std::size_t> m_cellRawIds;
    ScalarStorage<std::size_t> m_internalCellRawIds;

    ScalarStorage<std::size_t> m_interfaceRawIds;
    ScalarStorage<std::size_t> m_interfaceOwnerRawIds;
    ScalarStorage<std::size_t> m_interfaceNeighRawIds;

    using bitpit::PatchInfo::setPatch;

    virtual void _init();
    virtual void _reset();
    virtual void _extract();

private:
    ScalarPiercedStorage<double> m_cellVolumes;
    ScalarPiercedStorage<double> m_cellSizes;
    VectorPiercedStorage<double> m_cellCentroids;

    ScalarPiercedStorage<double> m_interfaceAreas;
    VectorPiercedStorage<double> m_interfaceCentroids;
    VectorPiercedStorage<double> m_interfaceNormals;
    VectorPiercedStorage<double> m_interfaceTangents;

};

#endif
