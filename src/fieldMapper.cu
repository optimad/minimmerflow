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

#include "fieldMapper.hcu"

namespace fieldMapper {

/*
 * Map solution from previous mesh to new one on GPU
 *
 * \param ids of pre-adaptation mesh
 * \param ids of post-adaptation mesh
 * \param[out] mapped field (on post-adaptation mesh)
 */
void cuda_mapField(std::size_t *previousIDs, std::size_t *currentIDs, double *field)
{
//    PiercedVector<double, long> previousField;
//    for (long previousId : previousIDs) {
////      previousField.insert(previousId, std::move(field.at(previousId)));
//    }
//    for (long currentId : currentIDs) {
////      field.set(currentId, previousField.at(parentId));
//    }
}

}
