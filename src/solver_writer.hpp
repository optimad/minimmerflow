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

#ifndef __MINIMMERFLOW_SOLVER_STREAMER_HPP__
#define __MINIMMERFLOW_SOLVER_STREAMER_HPP__

#include "constants.hpp"
#include "containers.hpp"
#include "storage.hpp"

#include <bitpit_patchkernel.hpp>

class SolverWriter: public bitpit::VTKBaseStreamer {

public:
    SolverWriter(bitpit::VolumeKernel *mesh,
                 ScalarPiercedStorage<double> *primitives, ScalarPiercedStorage<double> *conservatives, ScalarPiercedStorage<double> *RHS,
                 ScalarPiercedStorage<bool> *solved);

    void flushData(std::fstream &stream, const std::string &name, bitpit::VTKFormat codex) override;

private:
    static const int SOURCE_PRIMITIVE    = 0;
    static const int SOURCE_CONSERVATIVE = 1;
    static const int SOURCE_RHS          = 2;
    static const int SOURCE_SOLVED       = 3;

    bitpit::VolumeKernel *m_mesh;
    ScalarPiercedStorage<double> *m_primitives;
    ScalarPiercedStorage<double> *m_conservatives;
    ScalarPiercedStorage<double> *m_RHS;
    ScalarPiercedStorage<bool> *m_solved;

};

#endif
