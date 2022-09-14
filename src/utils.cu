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

#include "utils.hcu"

namespace utils {

/*!
 * Compute the factorial of the specified number.
 *
 * \param[in] n is the argument for which the factorial has to be evaluated
 * \result The factorial of the specified number.
 */
CUDA_HOST_DEVICE unsigned long factorial(unsigned long n)
{
    unsigned long factorial = 1;
    for (unsigned long i = 1; i <= n; ++i) {
        factorial *= i;
    }

    return factorial;
}

}
