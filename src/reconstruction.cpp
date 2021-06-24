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

#include "reconstruction.hpp"

using namespace bitpit;

namespace reconstruction {

/*!
 * Initialize the reconstructions.
 */
void initialize()
{
}

/*!
 * Compute the reconstruction polynomials.
 *
 * \param problemType is the problem type
 * \param computationInfo are the computation information
 * \param conservativeFields are the conservative fields
 * \param solvedBoundaryInterfaceBCs is the storage for the interface boundary
 * conditions of the solved boundary cells
 */
void computePolynomials(problem::ProblemType problemType, const ComputationInfo &computationInfo,
                       const ScalarPiercedStorageCollection<double> &conservativeFields, const ScalarStorage<int> &solvedBoundaryInterfaceBCs)
{
    BITPIT_UNUSED(problemType);
    BITPIT_UNUSED(computationInfo);
    BITPIT_UNUSED(conservativeFields);
    BITPIT_UNUSED(solvedBoundaryInterfaceBCs);
}

/*!
 * Evaluate the reconstruction are the specified point.
 *
 * \param order is the order
 * \param point is the point where the reconstruction will be evaluated
 * \param means are the mean values of the fields
 * \param[out] values on output will contain the reconstructed values
 */
void eval(int order, const std::array<double, 3> &point, const double *means, double *values)
{
    switch (order) {

    case (1):
        eval_1(point, means, values);
        break;

    default:
        exit(2);

    }
}

/*!
 * Evaluate the reconstruction are the specified point.
 *
 * \param order is the order
 * \param point is the point where the reconstruction will be evaluated
 * \param means are the mean values of the fields
 * \param[out] values on output will contain the reconstructed values
 */
void eval_1(const std::array<double, 3> &point, const double *means, double *values)
{
    BITPIT_UNUSED(point);

    std::copy_n(means, N_FIELDS, values);
}

}
