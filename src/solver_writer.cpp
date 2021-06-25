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

#include <bitpit_common.hpp>

#include "solver_writer.hpp"

using namespace bitpit;

/*!
 * Constructor
 *
 * \param mesh is the mesh
 * \param primitives is the storage with the primitive fields
 * \param conservatives is the storage with the conservatives fields
 * \param RHS is the storage with the RHS
 * \param solved is the storage with the solved flag
 */
SolverWriter::SolverWriter(VolumeKernel *mesh,
                           ScalarPiercedStorage<double> *primitives, ScalarPiercedStorage<double> *conservatives,
                           ScalarPiercedStorage<double> *RHS, ScalarPiercedStorage<int> *solved)
    : m_mesh(mesh),
      m_primitives(primitives), m_conservatives(conservatives), m_RHS(RHS),
      m_solved(solved)
{
}

/*!
 * Interface for VTK
 *
 * \param stream is the stream
 * \param name is the name of field to be written
 * \param codex is the format
 */
void SolverWriter::flushData(std::fstream &stream, const std::string &name, bitpit::VTKFormat codex)
{
    BITPIT_UNUSED(codex);

    int count  = 1;
    int offset = 0;
    int source = SOURCE_PRIMITIVE;
    if (name == "velocity") {
        count  = 3;
        offset = FID_U;
    } else if (name == "pressure") {
        offset = FID_P;
    } else if (name == "temperature") {
        offset = FID_T;
    } else if (name == "density") {
        offset = FID_RHO;
        source = SOURCE_CONSERVATIVE;
    } else if (name == "residualC") {
        offset = FID_EQ_C;
        source = SOURCE_RHS;
    } else if( name == "residualMX") {
        offset = FID_EQ_M_X;
        source = SOURCE_RHS;
    } else if( name == "residualMY") {
        offset = FID_EQ_M_Y;
        source = SOURCE_RHS;
    } else if( name == "residualMZ") {
        offset = FID_EQ_M_Z;
        source = SOURCE_RHS;
    } else if( name == "residualE") {
        offset = FID_EQ_E;
        source = SOURCE_RHS;
    } else if( name == "solved") {
        source = SOURCE_SOLVED;
    } else {
        return;
    }

    if (source == SOURCE_PRIMITIVE) {
        for (const Cell &cell : m_mesh->getCells()) {
            for (int k = 0; k < count; ++k) {
                double value = m_primitives->at(cell.getId(), offset + k);
                bitpit::genericIO::flushBINARY(stream, value);
            }
        }
    } else if (source == SOURCE_CONSERVATIVE) {
        for (const Cell &cell : m_mesh->getCells()) {
            for (int k = 0; k < count; ++k) {
                double value = m_conservatives->at(cell.getId(), offset + k);
                bitpit::genericIO::flushBINARY(stream, value);
            }
        }
    } else if (source == SOURCE_RHS) {
        for (const Cell &cell : m_mesh->getCells()) {
            double value = m_RHS->at(cell.getId(), offset);
            bitpit::genericIO::flushBINARY(stream, value);
        }
    } else if (source == SOURCE_SOLVED) {
        for (const Cell &cell : m_mesh->getCells()) {
            int value = m_solved->at(cell.getId(), offset) ? 1 : 0;
            bitpit::genericIO::flushBINARY(stream, value);
        }
    }
}
