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
    MeshGeometricalInfo(bitpit::VolumeKernel *patch);

    void setPatch(bitpit::VolumeKernel const *patch);
    bitpit::VolumeKernel const & getPatch() const;

    int getDimension() const;

    float getCellVolume(long id) const;
    float rawGetCellVolume(size_t pos) const;
    const bitpit::PiercedStorage<float, long> & getCellVolumes() const;
    bitpit::PiercedStorage<float, long> & getCellVolumes();
#if ENABLE_CUDA
    float * cuda_getCellVolumeDevData();
    const float * cuda_getCellVolumeDevData() const;
#endif

    float getCellSize(long id) const;
    float rawGetCellSize(size_t pos) const;
    const bitpit::PiercedStorage<float, long> & getCellSizes() const;
    bitpit::PiercedStorage<float, long> & getCellSizes();
#if ENABLE_CUDA
    float * cuda_getCellSizeDevData();
    const float * cuda_getCellSizeDevData() const;
#endif

    const std::array<float, 3> & getCellCentroid(long id) const;
    const std::array<float, 3> & rawGetCellCentroid(size_t pos) const;
    const bitpit::PiercedStorage<std::array<float, 3>, long> & getCellCentroids() const;
    bitpit::PiercedStorage<std::array<float, 3>, long> & getCellCentroids();
#if ENABLE_CUDA
    float * cuda_getCellCentroidDevData();
    const float * cuda_getCellCentroidDevData() const;
#endif

    float getInterfaceArea(long id) const;
    float rawGetInterfaceArea(size_t pos) const;
    const bitpit::PiercedStorage<float, long> & getInterfaceAreas() const;
    bitpit::PiercedStorage<float, long> & getInterfaceAreas();
#if ENABLE_CUDA
    float * cuda_getInterfaceAreaDevData();
    const float * cuda_getInterfaceAreaDevData() const;
#endif

    const std::array<float, 3> & getInterfaceCentroid(long id) const;
    const std::array<float, 3> & rawGetInterfaceCentroid(size_t pos) const;
    const bitpit::PiercedStorage<std::array<float, 3>, long> & getInterfaceCentroids() const;
    bitpit::PiercedStorage<std::array<float, 3>, long> & getInterfaceCentroids();
#if ENABLE_CUDA
    float * cuda_getInterfaceCentroidDevData();
    const float * cuda_getInterfaceCentroidDevData() const;
#endif

    const std::array<float, 3> & getInterfaceNormal(long id) const;
    const std::array<float, 3> & rawGetInterfaceNormal(size_t pos) const;
    const bitpit::PiercedStorage<std::array<float, 3>, long> & getInterfaceNormals() const;
    bitpit::PiercedStorage<std::array<float, 3>, long> & getInterfaceNormals();
#if ENABLE_CUDA
    float * cuda_getInterfaceNormalDevData();
    const float * cuda_getInterfaceNormalDevData() const;
#endif

    const std::array<float, 3> & getInterfaceTangent(long id) const;
    const std::array<float, 3> & rawGetInterfaceTangent(size_t pos) const;
    const bitpit::PiercedStorage<std::array<float, 3>, long> & getInterfaceTangents() const;
    bitpit::PiercedStorage<std::array<float, 3>, long> & getInterfaceTangents();
#if ENABLE_CUDA
    float * cuda_getInterfaceTangentDevData();
    const float * cuda_getInterfaceTangentDevData() const;
#endif

#if ENABLE_CUDA
    virtual void cuda_initialize();
    virtual void cuda_finalize();
#endif

protected:
    const bitpit::VolumeKernel *m_volumePatch;

    MeshGeometricalInfo(bitpit::VolumeKernel *patch, bool extractInfo);

    using bitpit::PatchInfo::setPatch;

    void _init() override;
    void _reset() override;
    void _extract() override;

private:
    ScalarPiercedStorage<float> m_cellVolumes;
    ScalarPiercedStorage<float> m_cellSizes;
    VectorPiercedStorage<float> m_cellCentroids;

    ScalarPiercedStorage<float> m_interfaceAreas;
    VectorPiercedStorage<float> m_interfaceCentroids;
    VectorPiercedStorage<float> m_interfaceNormals;
    VectorPiercedStorage<float> m_interfaceTangents;

};

#endif
