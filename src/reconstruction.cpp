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

#include "polynomials.hpp"
#include "reconstruction.hpp"

#include <bitpit_patchkernel.hpp>

#include <cassert>

using namespace bitpit;

/*!
 * \brief The ReconstructionCalculator class allows to eval cell
 * reconstructions.
 */

/*!
 * Creates a reconstruction calculator.
 *
 * \param computationInfo are the computation information
 * \param order is the order of the reconstruction
 */
ReconstructionCalculator::ReconstructionCalculator(const ComputationInfo &computationInfo, int order)
    : m_computationInfo(computationInfo),
      m_order(order), m_dimension(m_computationInfo.getPatch().getDimension()),
      m_nBasis(BasePolynomial::countBasis(m_dimension, m_order - 1)),
      m_cellPolynomials(N_FIELDS, m_nBasis)
{
    const VolumeKernel &patch = m_computationInfo.getPatch();
    const bitpit::PiercedVector<bitpit::Cell> cells = patch.getCells();

    // Set cell kernels
    m_cellSupportOffsets.setStaticKernel(&cells);
    m_cellSupportSizes.setStaticKernel(&cells);

    for (int k = 0; k < N_FIELDS; ++k) {
        m_cellPolynomials[k].setStaticKernel(&cells);
    }

    // Initialize reconstruction
    const ScalarStorage<std::size_t> &reconstructedCellRawIds = m_computationInfo.getReconstructedCellRawIds();
    const std::size_t nReconstructedCells = reconstructedCellRawIds.size();

    std::size_t previousSupportEnd = 0;
    for (std::size_t i = 0; i < nReconstructedCells; ++i) {
        // Cell information
        const std::size_t cellRawId = reconstructedCellRawIds[i];

        // Evaluate reconstruction support
        std::size_t *cellSupportSize   = m_cellSupportSizes.rawData(cellRawId);
        std::size_t *cellSupportOffset = m_cellSupportOffsets.rawData(cellRawId);

        evalCellSupport(cellRawId, cellSupportSize);
        *cellSupportOffset = previousSupportEnd;
        previousSupportEnd += *cellSupportSize;
    }
    m_cellSupportRawIds.resize(previousSupportEnd);
    m_cellKernelWeights.resize(m_nBasis * previousSupportEnd);

    ReconstructionKernel reconstructionKernel;
    for (std::size_t i = 0; i < nReconstructedCells; ++i) {
        // Cell information
        const std::size_t cellRawId = reconstructedCellRawIds[i];

        // Evaluate reconstruction support
        std::size_t cellSupportSize   = m_cellSupportSizes.rawAt(cellRawId);
        std::size_t cellSupportOffset = m_cellSupportOffsets.rawAt(cellRawId);
        long *cellSupportRawIds = m_cellSupportRawIds.data() + cellSupportOffset;
        std::size_t dummySupportSize;
        evalCellSupport(cellRawId, &dummySupportSize, cellSupportRawIds);

        // Evaluate reconstruction kernel
        double *cellKernelWeights = m_cellKernelWeights.data() + m_nBasis * cellSupportOffset;
        evalCellKernel(cellRawId, cellSupportSize, cellSupportRawIds, &reconstructionKernel);
        double *reconstructionWeightsBegin = reconstructionKernel.getPolynomialWeights();
        double *reconstructionWeightsEnd   = reconstructionWeightsBegin + (m_nBasis * cellSupportSize);
        std::copy(reconstructionWeightsBegin, reconstructionWeightsEnd, cellKernelWeights);
    }
}

/*!
 * Gets the order of the reconstruction.
 *
 * \result The order of the reconstruction
 */
int ReconstructionCalculator::getOrder() const
{
    return m_order;
}

/*!
 * Gets the dimension of the space.
 *
 * \result The dimension of the spcace
 */
int ReconstructionCalculator::getDimension() const
{
    return m_dimension;
}

/*!
 * Gets a constant reference to the cell polynomial storage.
 *
 * \result A constant reference to the cell polynomial storage.
 */
const ScalarPiercedStorageCollection<double> & ReconstructionCalculator::getCellPolynomials() const
{
    return m_cellPolynomials;
}

/*!
 * Gets a reference to the cell polynomial storage.
 *
 * \result A reference to the cell polynomial storage.
 */
ScalarPiercedStorageCollection<double> & ReconstructionCalculator::getCellPolynomials()
{
    return m_cellPolynomials;
}

/*!
 * Update the reconstructions.
 *
 * \param cellConservatives are the cell conservative values
 */
void ReconstructionCalculator::update(const ScalarPiercedStorageCollection<double> &cellConservatives)
{
    // Update reconstruction polynomials
#if ENABLE_CUDA
    cuda_updateCellPolynomials(cellConservatives);
#else
    updateCellPolynomials(cellConservatives);
#endif
}

#if ENABLE_CUDA==0
/*!
 * Updates cell reconstruction polynomials.
 *
 * \param cellConservatives are the cell conservative values
 */
void ReconstructionCalculator::updateCellPolynomials(const ScalarPiercedStorageCollection<double> &cellConservatives)
{
    // Evaluate polynomial coefficients
    const ScalarStorage<std::size_t> &reconstructedCellRawIds = m_computationInfo.getReconstructedCellRawIds();
    const std::size_t nReconstructedCells = reconstructedCellRawIds.size();

    typedef PolynomialCoefficientsCursor CellPolynomials;
    CellPolynomials cellPolynomialCursor(&m_cellPolynomials);

    PolynomialSupportFields supportCellConservatives(&cellConservatives);
    PolynomialAssembler polynomialAssembler(m_dimension, m_order - 1);

    for (std::size_t i = 0; i < nReconstructedCells; ++i) {
        // Cell information
        const std::size_t cellRawId = reconstructedCellRawIds[i];

        // Get support
        std::size_t cellSupportSize   = m_cellSupportSizes.rawAt(cellRawId);
        std::size_t cellSupportOffset = m_cellSupportOffsets.rawAt(cellRawId);
        const long *cellSupportRawIds = m_cellSupportRawIds.data() + cellSupportOffset;

        // Get reconstruction kernel information
        const double *cellWeights = m_cellKernelWeights.data() + m_nBasis * cellSupportOffset;

        // Assemble reconstruction polynomial
        cellPolynomialCursor.rawSet(cellRawId);
        polynomialAssembler.assemble(cellWeights, cellSupportSize, cellSupportRawIds,
                                     supportCellConservatives, &cellPolynomialCursor);
    }
}
#endif

/*!
 * Evaluate the reconstruction support for the specified cell.
 *
 * \param cellRawId is the raw id of the cell
 * \param supportSize on output will contain the support size
 * \param supportRawIds if a valid pointer is provided, on output will contain the support raw ids
 */
void ReconstructionCalculator::evalCellSupport(std::size_t cellRawId, std::size_t *supportSize, long *supportRawIds) const
{
    const VolumeKernel &patch = m_computationInfo.getPatch();
    const PiercedVector<Cell> &cells = patch.getCells();
    const PiercedVector<Interface> &interfaces = patch.getInterfaces();

    // Cell info
    const VolumeKernel::CellConstIterator cellItr = cells.rawFind(cellRawId);
    const Cell &cell = *cellItr;

    // Initialize support
    *supportSize = 0;

    // Add cell contribution
    if (supportRawIds) {
        supportRawIds[*supportSize] = cellItr.getRawIndex();
    }
    ++(*supportSize);

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
        if (supportRawIds) {
            long supportId = interface.getOwner();
            if (supportId == cellItr.getId()) {
                supportId = interface.getNeigh();
            }
            std::size_t supportRawId = cells.find(supportId).getRawIndex();

            supportRawIds[*supportSize] = supportRawId;
        }
        ++(*supportSize);

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
    const VolumeKernel &patch = m_computationInfo.getPatch();
    const PiercedVector<Cell> &cells = patch.getCells();

    // Cell info
    const VolumeKernel::CellConstIterator cellItr = patch.getCells().rawFind(cellRawId);
    const std::array<double, 3> &cellCentroid = m_computationInfo.rawGetCellCentroid(cellRawId);

    // Initialize assembler
    static ReconstructionAssembler assembler;
    assembler.initialize(m_order - 1, m_dimension, false);

    // Add neighbour contributions
    VolumeKernel::CellConstIterator supportItr;
    static std::vector<std::array<double BITPIT_COMMA 3>> vertexCoords;
    for (std::size_t i = 0; i < supportSize; ++i) {
        long supportId = support[i];
        if (supportId != cellItr.getId()) {
            supportItr = cells.find(supportId);
        } else {
            supportItr = cells.rawFind(supportId);
        }

        ConstProxyVector<long> vertexIds = supportItr->getVertexIds();
        std::size_t nVertices = vertexIds.size();
        vertexCoords.resize(nVertices);
        patch.getVertexCoords(nVertices, vertexIds.data(), vertexCoords.data());

        if (supportId != cellItr.getId()) {
            const std::array<double, 3> supportCentroid = m_computationInfo.rawGetCellCentroid(supportItr.getRawIndex());
            double equationWeight = 1. / std::pow(norm2(cellCentroid - supportCentroid), 3);
            assembler.addCellAverageEquation(Reconstruction::TYPE_LEAST_SQUARE, *supportItr, cellCentroid, vertexCoords.data(), equationWeight);
        } else {
            assembler.addCellAverageEquation(Reconstruction::TYPE_CONSTRAINT, *supportItr, cellCentroid, vertexCoords.data());
        }
    }

    // Assembly kernel
    assembler.assembleKernel(kernel);
}
