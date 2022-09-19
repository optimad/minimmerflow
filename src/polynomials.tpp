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

#ifndef __MINIMMERFLOW_POLYNOMIALS_TPP__
#define __MINIMMERFLOW_POLYNOMIALS_TPP__

/*!
 * Assemble the specified polynomial.
 *
 * \param weights are the geometrical weights that defines the polynomial
 * \param supportRawIds are the raw ids of the cells that define the support of the polynomial
 * \param supportFields are the fields on the cells that define the support of the polynomial
 * \param coefficients are the coefficients of the polynomial
 * \param[out] coefficients on output will contain the coefficients of the
 * polynomial
 */
template<typename Fields, typename Coefficients>
CUDA_HOST_DEVICE void PolynomialAssembler::assemble(const double *weights, int supportSize, const long *supportRawIds,
                                                    const Fields &supportFields, Coefficients *coefficients) const
{
    for (int k = 0; k < N_FIELDS; ++k) {
        double *fieldPolynomial = coefficients->data(k);

        const double *weightItr = weights;
        for (int i = 0; i < m_nBasis; ++i) {
            fieldPolynomial[i] = *weightItr * supportFields.rawAt(supportRawIds[0], k);
            ++weightItr;

            for (int j = 1; j < supportSize; ++j) {
                fieldPolynomial[i] += *weightItr * supportFields.rawAt(supportRawIds[j], k);
                ++weightItr;
            }
        }
    }
}

/*!
 * Evaluate the specified polynomial at the given point.
 *
 * \param origin is the origin of the polynomial
 * \param coefficients are the coefficients of the polynomial
 * \param point is the point where the polynomial will be evaluated
 * \param[out] values on output will contain the values of the polynomial
 */
template<typename Coefficients, typename Values>
CUDA_HOST_DEVICE void PolynomialCalculator::eval(const double *origin, const Coefficients &coefficients,
                                                 const double *point, Values *values) const
{
    // Constant polynomial
    double csi_0 = evalBasis(0, origin, point);
    for (int k = 0; k < N_FIELDS; ++k) {
        (*values)[k] = coefficients.data(k)[0] * csi_0;
    }

    if (m_nBasis == 1) {
        return;
    }

    // Generic polynomial
    for (int i = 1; i < m_nBasis; ++i) {
        double csi_i = evalBasis(i, origin, point);
        for (int k = 0; k < N_FIELDS; ++k) {
            (*values)[k] += coefficients.data(k)[i] * csi_i;
        }
    }
}

#endif
