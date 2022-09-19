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

#if ENABLE_CUDA
#include "compiler.hpp"
#include "polynomials.hcu"
#include "utils.hcu"
#else
#include "compiler.hpp"
#include "polynomials.hpp"
#include "utils.hpp"
#endif

#if ENABLE_CUDA
/*!
 * \brief The DevicePolynomialCoefficientsCursor allows to access polynomial coefficients
 * on the device.
 */

/*!
 * Constructor
 *
 * \param storage is a pointer to the storage
 * \param blockSize is the size of the coefficient blocks (each block of coefficients is
 * associated to a different cell)
 */
__device__ DevicePolynomialCoefficientsCursor::DevicePolynomialCoefficientsCursor(double **storage, std::size_t blockSize)
    : m_storage(storage), m_blockSize(blockSize)
{
}

/*!
 * Set the cursor.
 *
 * \param rawId is the raw id of the cell the cursor will point to
 */
__device__ void DevicePolynomialCoefficientsCursor::rawSet(std::size_t rawId)
{
    m_offset = m_blockSize * rawId;
}

/*!
 * Get a pointer to the coefficients of the specified field.
 *
 * \param field is the field
 * \result A pointer to the coefficients for the specified field.
 */
__device__ double * DevicePolynomialCoefficientsCursor::data(int field)
{
    return const_cast<double *>(static_cast<const DevicePolynomialCoefficientsCursor &>(*this).data(field));
}

/*!
 * Get a constant pointer to the coefficients of the specified field.
 *
 * \param field is the field
 * \result A constant pointer to the coefficients for the specified field.
 */
__device__ const double * DevicePolynomialCoefficientsCursor::data(int field) const
{
    return (m_storage[field] + m_offset);
}

/*!
 * \brief The DevicePolynomialCoefficientsConstCursor allows to access polynomial coefficients
 * on the device.
 */

/*!
 * Constructor
 *
 * \param storage is a pointer to the storage
 * \param blockSize is the size of the coefficient blocks (each block of coefficients is
 * associated to a different cell)
 */
__device__ DevicePolynomialCoefficientsConstCursor::DevicePolynomialCoefficientsConstCursor(const double * const *storage, std::size_t blockSize)
    : m_storage(storage), m_blockSize(blockSize)
{
}

/*!
 * Set the cursor.
 *
 * \param rawId is the raw id of the cell the cursor will point to
 */
__device__ void DevicePolynomialCoefficientsConstCursor::rawSet(std::size_t rawId)
{
    m_offset = m_blockSize * rawId;
}

/*!
 * Get a constant pointer to the coefficients of the specified field.
 *
 * \param field is the field
 * \result A constant pointer to the coefficients for the specified field.
 */
__device__ const double * DevicePolynomialCoefficientsConstCursor::data(int field) const
{
    return (m_storage[field] + m_offset);
}

/*!
 * \brief The DevicePolynomialSupportFields allows to access, on the device, to the fields
 * associated to the cells that define the support of a polynomial.
 */

/*!
 * Constructor
 *
 * \param storage is a pointer to the storage
 */
__device__ DevicePolynomialSupportFields::DevicePolynomialSupportFields(const double * const *storage)
    : m_storage(storage)
{
}

/*!
 * Get a constant pointer to the fields of the specified cell.
 *
 * \param rawId is the raw id of the cell
 * \param field is the field
 * \result A constant pointer to the fields of the specified cell.
 */
__device__ double DevicePolynomialSupportFields::rawAt(std::size_t rawId, int field) const
{
    return m_storage[field][rawId];
}
#endif

/*!
 * \brief The BasePolynomial is the base class to defining objects that
 * handles reconstruction polynomial.
 */

/*!
 * Count the basis that define the specified polynomial.
 *
 * \param dimension is the dimension of the space
 * \param degree is the degree of the polynomial
 * \result The number of basis that define the specified polynomial.
 */
CUDA_HOST_DEVICE int BasePolynomial::countBasis(uint8_t dimension, uint8_t degree)
{
    uint16_t nBasis = 0;
    for (int i = 0; i <= degree; ++i) {
        nBasis += (utils::factorial(dimension - 1 + i) / utils::factorial(dimension - 1) / utils::factorial(i));
    }

    return nBasis;
}

/*!
 * Constructor
 *
 * \param dimension is the dimension of the space
 * \param degree is the degree of the polynomial
 */
CUDA_HOST_DEVICE BasePolynomial::BasePolynomial(uint8_t dimension, uint8_t degree)
    : m_dimension(dimension), m_nBasis(BasePolynomial::countBasis(m_dimension, degree))
{
}

/*!
 * Evaluates the requested polynmial basis are the specified point.
 *
 * \param n is the index of the requested coefficient
 * \param origin is the origin of the polynomial
 * \param point the polynmial basis are the specified point.
 */
CUDA_HOST_DEVICE double BasePolynomial::evalBasis(int n, const double *origin, const double *point) const
{
    assert(n <m_nBasis);

    // 0-th degree basis
    if (n == 0) {
        return 1.;
    }

    // 1-st degree bases
    if (n < (1 + m_dimension)) {
        int coord       = n - 1;
        double distance = point[coord] - origin[coord];

        return distance;
    }

    // Set 2-nd degree bases
    if (n < (1 + 2 * m_dimension)) {
        int coord = n - (1 + m_dimension);

        double distance = point[coord] - origin[coord];

        return 0.5 * distance * distance;
    }

    if (n < (1 + 2 * m_dimension + 2)) {
        int coord_a = n - (1 + 2 * m_dimension);
        int coord_b = 2;

        double distance_a = point[coord_a] - origin[coord_a];
        double distance_b = point[coord_b] - origin[coord_b];

        return distance_a * distance_b;
    }

    return 0;
}


/*!
 * \brief The PolynomialAssembler class allows to assemble a reconstruction
 * polynomial.
 */

/*!
 * Constructor
 *
 * \param dimension is the dimension of the space
 * \param degree is the degree of the polynomial
 */
CUDA_HOST_DEVICE PolynomialAssembler::PolynomialAssembler(uint8_t dimension, uint8_t degree)
    : BasePolynomial(dimension, degree)
{
}

/*!
 * \brief The PolynomialCalculator class allows to evaluate a reconstruction
 * polynomial.
 */

/*!
 * Constructor
 *
 * \param dimension is the dimension of the space
 * \param degree is the degree of the polynomial
 */
CUDA_HOST_DEVICE PolynomialCalculator::PolynomialCalculator(uint8_t dimension, uint8_t degree)
    : BasePolynomial(dimension, degree)
{
}
