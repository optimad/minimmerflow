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

    double getCellCentroid(long id, int component) const;
    std::array<double, 3> getCellCentroid(long id) const;
    double rawGetCellCentroid(size_t pos, int component) const;
    std::array<double, 3> rawGetCellCentroid(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getCellCentroids(int component) const;
    bitpit::PiercedStorage<double, long> & getCellCentroids(int component);
#if ENABLE_CUDA
    double ** cuda_getCellCentroidDevData();
    const double * const * cuda_getCellCentroidDevData() const;
#endif

    double getInterfaceArea(long id) const;
    double rawGetInterfaceArea(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getInterfaceAreas() const;
    bitpit::PiercedStorage<double, long> & getInterfaceAreas();
#if ENABLE_CUDA
    double * cuda_getInterfaceAreaDevData();
    const double * cuda_getInterfaceAreaDevData() const;
#endif

    double getInterfaceCentroid(long id, int component) const;
    std::array<double, 3> getInterfaceCentroid(long id) const;
    double rawGetInterfaceCentroid(size_t pos, int component) const;
    std::array<double, 3> rawGetInterfaceCentroid(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getInterfaceCentroids(int component) const;
    bitpit::PiercedStorage<double, long> & getInterfaceCentroids(int component);
#if ENABLE_CUDA
    double ** cuda_getInterfaceCentroidDevData();
    const double * const * cuda_getInterfaceCentroidDevData() const;
#endif

    double getInterfaceNormal(long id, int component) const;
    std::array<double, 3> getInterfaceNormal(long id) const;
    double rawGetInterfaceNormal(size_t pos, int component) const;
    std::array<double, 3> rawGetInterfaceNormal(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getInterfaceNormals(int component) const;
    bitpit::PiercedStorage<double, long> & getInterfaceNormals(int component);
#if ENABLE_CUDA
    double ** cuda_getInterfaceNormalDevData();
    const double * const * cuda_getInterfaceNormalDevData() const;
#endif

    double getInterfaceTangent(long id, int component) const;
    std::array<double, 3> getInterfaceTangent(long id) const;
    double rawGetInterfaceTangent(size_t pos, int component) const;
    std::array<double, 3> rawGetInterfaceTangent(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getInterfaceTangents(int component) const;
    bitpit::PiercedStorage<double, long> & getInterfaceTangents(int component);
#if ENABLE_CUDA
    double ** cuda_getInterfaceTangentDevData();
    const double * const * cuda_getInterfaceTangentDevData() const;
#endif

#if ENABLE_CUDA
    virtual void cuda_initialize();
    void cuda_updateMeshInfo();
    virtual void cuda_finalize();
    virtual void cuda_resize();
#endif

protected:
    bitpit::VolumeKernel *m_volumePatch;

    MeshGeometricalInfo(bitpit::VolumeKernel *patch, bool extractInfo);

    using bitpit::PatchInfo::setPatch;

    void _init() override;
    void _reset() override;
    void _extract() override;

private:
    ScalarPiercedStorage<double> m_cellVolumes;
    ScalarPiercedStorage<double> m_cellSizes;
    ScalarPiercedStorageCollection<double> m_cellCentroids;

    ScalarPiercedStorage<double> m_interfaceAreas;
    ScalarPiercedStorageCollection<double> m_interfaceCentroids;
    ScalarPiercedStorageCollection<double> m_interfaceNormals;
    ScalarPiercedStorageCollection<double> m_interfaceTangents;

};

#endif
