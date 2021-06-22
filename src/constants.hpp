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

#ifndef __MINIMMERFLOW_CONSTANTS_HPP__
#define __MINIMMERFLOW_CONSTANTS_HPP__

// Constants
#define N_FIELDS 5

const int FID_P = 0;
const int FID_U = 1;
const int FID_V = 2;
const int FID_W = 3;
const int FID_T = 4;

const int FID_RHO   = 0;
const int FID_RHO_U = 1;
const int FID_RHO_V = 2;
const int FID_RHO_W = 3;
const int FID_RHO_E = 4;

const int FID_EQ_C   = 0;
const int FID_EQ_M_X = 1;
const int FID_EQ_M_Y = 2;
const int FID_EQ_M_Z = 3;
const int FID_EQ_E   = 4;

const float GAMMA = 1.4;
const float R     = 287;

const int BC_NONE       = -1;
const int BC_FREE_FLOW  =  0;
const int BC_REFLECTING =  1;
const int BC_WALL       =  2;
const int BC_DIRICHLET  =  3;

const int BC_INFO_SIZE = N_FIELDS;

const int SOLVE_SOLID_GHOST = -2;
const int SOLVE_SOLID       = -1;
const int SOLVE_UNDEFINED   =  0;
const int SOLVE_FLUID       =  1;
const int SOLVE_FLUID_GHOST =  2;

#if ENABLE_CUDA
namespace constants {

    void cuda_initialize();

}
#endif

#endif
