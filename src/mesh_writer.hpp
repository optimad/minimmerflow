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

#ifndef __MINIMMERFLOW_MESH_STREAMER_HPP__
#define __MINIMMERFLOW_MESH_STREAMER_HPP__

#include <bitpit_patchkernel.hpp>

#include "storage.hpp"

class MeshWriter: public bitpit::VTKBaseStreamer {

public:
    MeshWriter(bitpit::VolumeKernel *mesh, CellStorageDouble *primitives, CellStorageDouble *conservatives, CellStorageDouble *RHS);

    void flushData(std::fstream &stream, const std::string &name, bitpit::VTKFormat format) override;

private:
    static const int SOURCE_PRIMITIVE    = 0;
    static const int SOURCE_CONSERVATIVE = 1;
    static const int SOURCE_RHS          = 2;

    bitpit::VolumeKernel *m_mesh;
    CellStorageDouble *m_primitives;
    CellStorageDouble *m_conservatives;
    CellStorageDouble *m_RHS;

};

#endif
