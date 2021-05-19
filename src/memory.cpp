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

#include "memory.hpp"

#include <bitpit_IO.hpp>

using namespace bitpit;

/*!
 * Prints to log::cout() the status of memory usage by the current process
 *
 */
void log_memory_status()
{
  statm_t result;
//  unsigned long dummy;
  const char* statm_path = "/proc/self/statm";

  FILE *f = fopen(statm_path,"r");
  if(!f){
    perror(statm_path);
    abort();
  }
  if(7 != fscanf(f,"%ld %ld %ld %ld %ld %ld %ld",
    &result.size,&result.resident,&result.share,&result.text,&result.lib,&result.data,&result.dt))
  {
    perror(statm_path);
    abort();
  }
  fclose(f);

  log::cout() << " %%%%%%%%%%%%%%%%%%%%%%%%%%%%% " << std::endl;
  log::cout() << "   RAM usage :: ";
  log::cout() << " size     = " << result.size;
  log::cout() << " resident = " << result.resident;
  log::cout() << " data     = " << result.data << std::endl;
  log::cout() << " %%%%%%%%%%%%%%%%%%%%%%%%%%%%% " << std::endl;
}
