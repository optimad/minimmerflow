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

/*!
 * \brief The ReconstructionCalculator class allows to eval cell
 * reconstructions.
 */

/*!
 * Creates a reconstruction calculator.
 *
 * \param meshInfo are the mesh information
 * \param order is the order of the reconstruction
 */
ReconstructionCalculator::ReconstructionCalculator(const MeshGeometricalInfo &meshInfo, int order)
    : m_meshInfo(meshInfo), m_order(order)
{
    const VolumeKernel &patch = m_meshInfo.getPatch();
    const bitpit::PiercedVector<bitpit::Cell> cells = patch.getCells();

    // Set cell kernels
    m_cellSupports.setStaticKernel(&cells);
    m_cellKernels.setStaticKernel(&cells);
    m_cellPolynomials.setStaticKernel(&cells);

    // Initialize reconstruction
    for (VolumeKernel::CellConstIterator cellItr = patch.cellConstBegin(); cellItr != patch.cellConstEnd(); ++cellItr) {
        // Cell information
        std::size_t cellRawId = cellItr.getRawIndex();

        // Evaluate reconstruction support
        std::vector<long> *cellSupport = m_cellSupports.rawData(cellRawId);
        evalCellSupport(cellRawId, cellSupport);

        // Evaluate reconstruction kernel
        ReconstructionKernel *cellKernel = m_cellKernels.rawData(cellRawId);
        evalCellKernel(cellRawId, cellSupport->size(), cellSupport->data(), cellKernel);

        // Initialize reconstruction polynomial
        ReconstructionPolynomial *cellPolynomial = m_cellPolynomials.rawData(cellRawId);
        initializeCellPolynomial(cellRawId, cellPolynomial);
    }
}

/*!
 * Update the reconstructions.
 *
 * \param fields are the fields
 */
void ReconstructionCalculator::update(const CellStorageDouble &fields)
{
    const VolumeKernel &patch = m_meshInfo.getPatch();

    for (VolumeKernel::CellConstIterator cellItr = patch.cellConstBegin(); cellItr != patch.cellConstEnd(); ++cellItr) {
        // Cell information
        std::size_t cellRawId = cellItr.getRawIndex();

        // Update reconstruction polynomial
        ReconstructionPolynomial *cellPolynomial = m_cellPolynomials.rawData(cellRawId);
        updateCellPolynomial(cellRawId, fields, cellPolynomial);
    }
}

/*!
 * Evaluate cell values at the specified point.
 *
 * \param cellRawId is the raw id of the cell
 * \param point is the point where the values will be evaluated
 * \param[out] values on output will contain the reconstructed values
 */
void ReconstructionCalculator::evalCellValues(std::size_t cellRawId, const std::array<double, 3> &point, double *values) const
{
    const ReconstructionPolynomial &polynomial = m_cellPolynomials.rawAt(cellRawId);
    evalPolynomial(polynomial, point, values);
}

/*!
 * Evaluate the reconstruction support for the specified cell.
 *
 * \param cellRawId is the raw id of the cell
 * \param support on output will contain the support
 */
void ReconstructionCalculator::evalCellSupport(std::size_t cellRawId, std::vector<long> *support) const
{
    const VolumeKernel &patch = m_meshInfo.getPatch();
    const PiercedVector<Cell> &cells = patch.getCells();
    const PiercedVector<Interface> &interfaces = patch.getInterfaces();

    // Cell info
    const VolumeKernel::CellConstIterator cellItr = cells.rawFind(cellRawId);
    const Cell &cell = *cellItr;

    // Initialize support
    support->clear();

    // Add cell contribution
    support->push_back(cellItr.getId());

    // Add neighbour contributions
    const int nCellInterfaces = cell.getInterfaceCount();
    const long *cellInterfaces = cell.getInterfaces();
    for (int k = 0; k < nCellInterfaces; ++k) {
        // Interface info
        long interfaceId = cellInterfaces[k];
        const Interface &interface = interfaces.at(interfaceId);

        // Discard boundary interfaces
        if (interface.getNeigh() < 0) {
            continue;
        }

        // Add neighbour
        long neighId = interface.getOwner();
        if (neighId == cellItr.getId()) {
            neighId = interface.getNeigh();
        }

        support->push_back(neighId);
    }
}

/*!
 * Evaluate the reconstruction kernel for the specified cell.
 *
 * \param cellRawId is the raw id of the cell
 * \param supportSize is the size of the support
 * \param support is the support of the cell
 * \param kernel on output will contain the kernel
 */
void ReconstructionCalculator::evalCellKernel(std::size_t cellRawId, std::size_t supportSize, const long *support,
                                              ReconstructionKernel *kernel) const
{
    const VolumeKernel &patch = m_meshInfo.getPatch();
    const PiercedVector<Cell> &cells = patch.getCells();

    // Cell info
    const VolumeKernel::CellConstIterator cellItr = patch.getCells().rawFind(cellRawId);
    const std::array<double, 3> &cellCentroid = m_meshInfo.rawGetCellCentroid(cellRawId);

    // Initialize assembler
    static ReconstructionAssembler assembler;
    assembler.initialize(m_order - 1, patch.getDimension(), false);

    // Add neighbour contributions
    VolumeKernel::CellConstIterator supportItr;
    static std::vector<std::array<double BITPIT_COMMA 3>> vertexCoords;
    for (std::size_t i = 0; i < supportSize; ++i) {
        long supportId = support[i];
        if (supportId != cellItr.getId()) {
            supportItr = cells.find(supportId);
        } else {
            supportItr = cells.rawFind(supportId);        }

        ConstProxyVector<long> vertexIds = supportItr->getVertexIds();
        std::size_t nVertices = vertexIds.size();
        vertexCoords.resize(nVertices);
        patch.getVertexCoords(nVertices, vertexIds.data(), vertexCoords.data());

        if (supportId != cellItr.getId()) {
            const std::array<double, 3> supportCentroid = m_meshInfo.rawGetCellCentroid(supportItr.getRawIndex());
            double equationWeight = 1. / std::pow(norm2(cellCentroid - supportCentroid), 3);
            assembler.addCellAverageEquation(Reconstruction::TYPE_LEAST_SQUARE, *supportItr, cellCentroid, vertexCoords.data(), equationWeight);
        } else {
            assembler.addCellAverageEquation(Reconstruction::TYPE_CONSTRAINT, *supportItr, cellCentroid, vertexCoords.data());
        }
    }

    // Assembly kernel
    assembler.assembleKernel(kernel);
}

/*!
 * Initializes the reconstruction polynomial of the specified cell.
 *
 * \param cellRawId is the raw id of the cell
 * \param[out] polynomial on output will contain the reconstruction
 * polynomial of the specified cell
 */
void ReconstructionCalculator::initializeCellPolynomial(std::size_t cellRawId, ReconstructionPolynomial *polynomial) const
{
    const VolumeKernel &patch = m_meshInfo.getPatch();
    int dimension = patch.getDimension();

    const std::array<double, 3> &cellCentroid = m_meshInfo.rawGetCellCentroid(cellRawId);

    polynomial->initialize(m_order - 1, dimension, cellCentroid, N_FIELDS, true);
}

/*!
 * Updates the reconstruction polynomial of the specified cell.
 *
 * \param cellRawId is the raw id of the cell
 * \param fields are the fields
 * \param[out] polynomial on output will contain the reconstruction
 * polynomial of the specified cell
 */
void ReconstructionCalculator::updateCellPolynomial(std::size_t cellRawId,
                                                    const CellStorageDouble &fields,
                                                    ReconstructionPolynomial *polynomial) const
{
    const VolumeKernel &patch = m_meshInfo.getPatch();
    const PiercedVector<Cell> &cells = patch.getCells();

    // Get support
    const std::vector<long> &cellSupport = m_cellSupports.rawAt(cellRawId);
    std::size_t cellSupportSize = cellSupport.size();

    // Get support values
    static std::vector<double> supportValuesStorage;
    supportValuesStorage.resize(cellSupportSize * N_FIELDS);

    static std::vector<const double *> supportValues;
    supportValues.resize(cellSupportSize, nullptr);

    for (std::size_t n = 0; n < cellSupportSize; ++n) {
        long supportId = cellSupport[n];
        std::size_t supportRawId = cells.find(supportId).getRawIndex();

        for (int k = 0; k < N_FIELDS; ++k) {
            supportValuesStorage[n * N_FIELDS + k] = fields.rawAt(supportRawId, k);
        }
        supportValues[n] = supportValuesStorage.data() + n * N_FIELDS;
    }

    // Get cell reconstruction kernel
    const ReconstructionKernel &kernel = m_cellKernels.rawAt(cellRawId);

    // Update reconstruction polynomial
    kernel.updatePolynomial(N_FIELDS, supportValues.data(), polynomial);
}

/*!
 * Evaluate the polynomial of the specified cell at the given point.
 *
 * \param polynomial is the polynomial to be evaluated
 * \param point is the point where the polynomial will be evaluated
 * \param[out] values on output will contain the values of the polynomial
 */
void ReconstructionCalculator::evalPolynomial(const ReconstructionPolynomial &polynomial,
                                              const std::array<double, 3> &point, double *values) const
{
    polynomial.computeValues(m_order - 1, point , N_FIELDS, 0, values);
}
