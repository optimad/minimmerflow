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

#ifndef __MINIMMERFLOW_POLYNOMIALS_HPP__
#define __MINIMMERFLOW_POLYNOMIALS_HPP__

#include "compiler.hpp"
#include "containers.hpp"

class PolynomialCoefficientsCursor
{

public:
    PolynomialCoefficientsCursor(ScalarPiercedStorageCollection<double> *storage);

    void rawSet(std::size_t rawId);

    double * data(int field);
    const double * data(int field) const;

private:
    ScalarPiercedStorageCollection<double> *m_storage;

    std::size_t m_rawId;

};

class PolynomialCoefficientsConstCursor
{

public:
    PolynomialCoefficientsConstCursor(const ScalarPiercedStorageCollection<double> *storage);

    void rawSet(std::size_t rawId);

    const double * data(int field) const;

private:
    const ScalarPiercedStorageCollection<double> *m_storage;

    std::size_t m_rawId;

};

class PolynomialSupportFields
{

public:
    PolynomialSupportFields(const ScalarPiercedStorageCollection<double> *storage);

    double rawAt(std::size_t rawId, int field) const;

private:
    const ScalarPiercedStorageCollection<double> *m_storage;

};

class BasePolynomial
{

public:
    CUDA_HOST_DEVICE static int countBasis(uint8_t dimensions, uint8_t degree);

    CUDA_HOST_DEVICE BasePolynomial(uint8_t dimension, uint8_t degree);

protected:
    uint8_t m_dimension;
    uint8_t m_nBasis;

    CUDA_HOST_DEVICE double evalBasis(int n, const double *origin, const double *point) const;

};

class PolynomialAssembler : public BasePolynomial
{

public:
    CUDA_HOST_DEVICE PolynomialAssembler(uint8_t dimension, uint8_t degree);

    template<typename Fields, typename Coefficients>
    CUDA_HOST_DEVICE void assemble(const double *weights, int supportSize, const long *supportRawIds,
                                   const Fields &supportFields, Coefficients *coefficients) const;

};

class PolynomialCalculator : public BasePolynomial
{

public:
    CUDA_HOST_DEVICE PolynomialCalculator(uint8_t dimension, uint8_t degree);

    template<typename Coefficients, typename Values>
    CUDA_HOST_DEVICE void eval(const double *origin, const Coefficients &coefficients,
                               const double *point, Values *values) const;

};

// Include template definitions
#include <polynomials.tpp>

#endif
