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

#include <polynomials.hpp>

#if ENABLE_CUDA==0
#include <polynomials.cu>
#endif

/*!
 * \brief The PolynomialCoefficientsCursor allows to access polynomial coefficients.
 */

/*!
 * Constructor
 *
 * \param storage is the storage that contains the coefficients
 */
PolynomialCoefficientsCursor::PolynomialCoefficientsCursor(ScalarPiercedStorageCollection<double> *storage)
    : m_storage(storage)
{
}

/*!
 * Set the cursor.
 *
 * \param rawId is the raw id of the cell the cursor will point to
 */
void PolynomialCoefficientsCursor::rawSet(std::size_t rawId)
{
    m_rawId = rawId;
}

/*!
 * Get a pointer to the coefficients of the specified field.
 *
 * \param field is the field
 * \result A pointer to the coefficients for the specified field.
 */
double * PolynomialCoefficientsCursor::data(int field)
{
    return const_cast<double *>(static_cast<const PolynomialCoefficientsCursor &>(*this).data(field));
}

/*!
 * Get a constant pointer to the coefficients of the specified field.
 *
 * \param field is the field
 * \result A constant pointer to the coefficients for the specified field.
 */
const double * PolynomialCoefficientsCursor::data(int field) const
{
    return (*m_storage)[field].rawData(m_rawId);
}

/*!
 * \brief The PolynomialCoefficientsCursor allows to access polynomial coefficients.
 */

/*!
 * Constructor
 *
 * \param storage is the storage that contains the coefficients
 */
PolynomialCoefficientsConstCursor::PolynomialCoefficientsConstCursor(const ScalarPiercedStorageCollection<double> *storage)
    : m_storage(storage)
{
}

/*!
 * Set the cursor.
 *
 * \param rawId is the raw id of the cell the cursor will point to
 */
void PolynomialCoefficientsConstCursor::rawSet(std::size_t rawId)
{
    m_rawId = rawId;
}

/*!
 * Get a constant pointer to the coefficients of the specified field.
 *
 * \param field is the field
 * \result A constant pointer to the coefficients for the specified field.
 */
const double * PolynomialCoefficientsConstCursor::data(int field) const
{
    return (*m_storage)[field].rawData(m_rawId);
}

/*!
 * \brief The PolynomialSupportFields allows to access to the fields associated
 * to the cells that define the support of a polynomial.
 */

/*!
 * Constructor
 *
 * \param storage is a pointer to the storage
 */
PolynomialSupportFields::PolynomialSupportFields(const ScalarPiercedStorageCollection<double> *storage)
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
double PolynomialSupportFields::rawAt(std::size_t rawId, int field) const
{
    return (*m_storage)[field].rawAt(rawId);
}
