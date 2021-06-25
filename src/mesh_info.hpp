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

#include <bitpit_containers.hpp>
#include <bitpit_patchkernel.hpp>

class MeshGeometricalInfo : public bitpit::PatchInfo {

public:
    MeshGeometricalInfo(bitpit::VolumeKernel *patch = nullptr);

    void setPatch(bitpit::VolumeKernel const *patch);
    bitpit::VolumeKernel const & getPatch() const;

    int getDimension() const;

    const std::vector<std::size_t> & getCellRawIds() const;
    const std::vector<std::size_t> & getInternalCellRawIds() const;

    double getCellVolume(long id) const;
    double rawGetCellVolume(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getCellVolumes() const;
    bitpit::PiercedStorage<double, long> & getCellVolumes();

    double getCellSize(long id) const;
    double rawGetCellSize(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getCellSizes() const;
    bitpit::PiercedStorage<double, long> & getCellSizes();

    const std::array<double, 3> & getCellCentroid(long id) const;
    const std::array<double, 3> & rawGetCellCentroid(size_t pos) const;
    const bitpit::PiercedStorage<std::array<double, 3>, long> & getCellCentroids() const;
    bitpit::PiercedStorage<std::array<double, 3>, long> & getCellCentroids();

    const std::vector<std::size_t> & getInterfaceRawIds() const;

    double getInterfaceArea(long id) const;
    double rawGetInterfaceArea(size_t pos) const;
    const bitpit::PiercedStorage<double, long> & getInterfaceAreas() const;
    bitpit::PiercedStorage<double, long> & getInterfaceAreas();

    const std::array<double, 3> & getInterfaceCentroid(long id) const;
    const std::array<double, 3> & rawGetInterfaceCentroid(size_t pos) const;
    const bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceCentroids() const;
    bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceCentroids();

    const std::array<double, 3> & getInterfaceNormal(long id) const;
    const std::array<double, 3> & rawGetInterfaceNormal(size_t pos) const;
    const bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceNormals() const;
    bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceNormals();

    const std::array<double, 3> & getInterfaceTangent(long id) const;
    const std::array<double, 3> & rawGetInterfaceTangent(size_t pos) const;
    const bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceTangents() const;
    bitpit::PiercedStorage<std::array<double, 3>, long> & getInterfaceTangents();

protected:
    const bitpit::VolumeKernel *m_volumePatch;

    std::vector<std::size_t> m_cellRawIds;
    std::vector<std::size_t> m_internalCellRawIds;

    std::vector<std::size_t> m_interfaceRawIds;

    using bitpit::PatchInfo::setPatch;

    virtual void _init();
    virtual void _reset();
    virtual void _extract();

private:
    bitpit::PiercedStorage<double, long> m_cellVolumes;
    bitpit::PiercedStorage<double, long> m_cellSizes;
    bitpit::PiercedStorage<std::array<double, 3>, long> m_cellCentroids;

    bitpit::PiercedStorage<double, long> m_interfaceAreas;
    bitpit::PiercedStorage<std::array<double, 3>, long> m_interfaceCentroids;
    bitpit::PiercedStorage<std::array<double, 3>, long> m_interfaceNormals;
    bitpit::PiercedStorage<std::array<double, 3>, long> m_interfaceTangents;

};

#endif
