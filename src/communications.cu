/*---------------------------------------------------------------------------*\
 *
 *  minimmerflow
 *
 *  Copyright (C) 2015-2022 OPTIMAD engineering Srl
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


#include "communications.hcu"
#include "communications.hpp"
#include "containers.hcu"

// Include template implementation
#include "communications.tcu"

#include "cuda_runtime_api.h"

// Explicit instantiation

template class CudaStorageBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>;
template class CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>>;


/*!
 * Send ghosts data using non-blocking communications
 *
 * \param cellData is the container of the cell data
 */
void ListCommunicator::startAllExchanges()
{
    if (getCommunicator() == MPI_COMM_NULL || !hasData()) {
        return;
    }

    // Start the receives
    for (int rank : getRecvRanks()) {
        if (!isRecvActive(rank)) {
            startRecv(rank);
        }
    }

    // Wait previous sends
    waitAllSends();

    // Fill the buffer with the given field and start sending the data
    for (int rank : getSendRanks()) {
        // Get send buffer
        bitpit::SendBuffer &buffer = getSendBuffer(rank);

        // Write the buffer
        for (ExchangeBufferStreamer *streamer : m_writers) {
            streamer->write(rank, buffer, getStreamableSendList(rank, streamer));
        }

        cudaDeviceSynchronize();

        // Start the send
        startSend(rank);
    }
}

void ListCommunicator::initializeCudaObjects()
{
    if (getCommunicator() == MPI_COMM_NULL || !hasData()) {
        return;
    }
    for (int rank : getSendRanks()) {
        bitpit::SendBuffer &buffer = getSendBuffer(rank);
        std::size_t bufferSize = buffer.getSize();
        size_t bytes = bufferSize * sizeof(char);
        cudaError_t err = cudaHostRegister(buffer.getFront().data(), bytes, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            std::cout << "CUDA runtime error in cudaHostRegister " << cudaGetErrorString(err) << " on buffer for rank " << rank << std::endl;
        }
    }
}
void ListCommunicator::finalizeCudaObjects()
{
    if (getCommunicator() == MPI_COMM_NULL || !hasData()) {
        return;
    }
    for (int rank : getSendRanks()) {
        bitpit::SendBuffer &buffer = getSendBuffer(rank);
        cudaError_t err = cudaHostUnregister(buffer.getFront().data());
        if (err != cudaSuccess) {
            std::cout << "CUDA runtime error in cudaHostUnregister " << cudaGetErrorString(err)  << " pointer " << (void *)buffer.getFront().data()
                    << " Comm " << m_name << std::endl;
        }
    }
}


ListCommunicator::~ListCommunicator()
{
}

OpenACCStreams::OpenACCStreams(int nFields)
{
    m_cudaStreams.resize(nFields);
    m_streamIds.resize(nFields, 0);
    for (int i = 0; i < nFields; ++i) {
        m_streamIds[i] = i;
        cudaStreamCreate(&(m_cudaStreams[i]));
        acc_set_cuda_stream(m_streamIds[i], m_cudaStreams[i]);
    }
}
OpenACCStreams::~OpenACCStreams()
{
    for (int i = 0; i < m_cudaStreams.size(); ++i) {
        cudaStreamDestroy(m_cudaStreams[i]);
    }
}
