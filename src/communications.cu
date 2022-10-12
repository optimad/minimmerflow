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
    //WARNING: the follow part works if only one writer has been added to the communicator
    CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>> * cudaStreamer =
            static_cast<CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>> *>(m_writers[0]);
    for (int rank : getSendRanks()) {
        bitpit::SendBuffer &buffer = getSendBuffer(rank);

        cudaStreamer->prepareWrite(rank, buffer, getStreamableSendList(rank, cudaStreamer));
        cudaStreamer->write(rank, buffer, getStreamableSendList(rank, cudaStreamer));
    }
    for (int rank : getSendRanks()) {
        cudaStreamSynchronize(cudaStreamer->m_queuesStreams.getCudaStreamByRank(rank));
        startSend(rank);
    }

}

void ListCommunicator::prepare1StartAllExchanges()
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
    //WARNING: the follow part works if only one writer has been added to the communicator
    CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>> * cudaStreamer =
            static_cast<CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>> *>(m_writers[0]);
    for (int rank : getSendRanks()) {
        bitpit::SendBuffer &buffer = getSendBuffer(rank);

        cudaStreamer->prepareWrite(rank, buffer, getStreamableSendList(rank, cudaStreamer));
    }
}

void ListCommunicator::prepare2StartAllExchanges()
{
    CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>> * cudaStreamer =
            static_cast<CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>> *>(m_writers[0]);
    for (int rank : getSendRanks()) {
        bitpit::SendBuffer &buffer = getSendBuffer(rank);

        cudaStreamer->write(rank, buffer, getStreamableSendList(rank, cudaStreamer));
    }
}

void ListCommunicator::completeStartAllExchanges()
{
    CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>> * cudaStreamer =
            static_cast<CudaStorageCollectionBufferStreamer<std::unordered_map<int, ScalarStorage<double>>> *>(m_writers[0]);
    for (int rank : getSendRanks()) {
        cudaStreamSynchronize(cudaStreamer->m_queuesStreams.getCudaStreamByRank(rank));
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
    for (int rank : getRecvRanks()) {
        bitpit::RecvBuffer &buffer = getRecvBuffer(rank);
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
            std::cout << "CUDA runtime error in cudaHostUnregister (SEND)" << cudaGetErrorString(err)  << " pointer " << (void *)buffer.getFront().data()
                    << " Comm " << m_name << std::endl;
        }
    }
    for (int rank : getRecvRanks()) {
        bitpit::RecvBuffer &buffer = getRecvBuffer(rank);
        cudaError_t err = cudaHostUnregister(buffer.getFront().data());
        if (err != cudaSuccess) {
            std::cout << "CUDA runtime error in cudaHostUnregister (RECV)" << cudaGetErrorString(err)  << " pointer " << (void *)buffer.getFront().data()
                    << " Comm " << m_name << std::endl;
        }
    }
}


ListCommunicator::~ListCommunicator()
{
}

QueuesStreams::QueuesStreams(const std::unordered_map<int, std::vector<long>> & sourceLists,
        const std::unordered_map<int, std::vector<long>> & targetLists, MPI_Comm communicator)
{
    std::set<int> sourceRanks, targetRanks;
    for (const auto & sourceList :sourceLists) {
        sourceRanks.insert(sourceList.first);
    }
    for (const auto & targetList :targetLists) {
        targetRanks.insert(targetList.first);
    }
    bool debug = sourceRanks == targetRanks;
    MPI_Allreduce(MPI_IN_PLACE, &debug, 1, MPI_CXX_BOOL, MPI_LAND, communicator);

    if (debug) {
        std::runtime_error("========================================================> Source ranks different from Target ranks! Cannot continue...");
    }


    for (int r : sourceRanks) {
        m_queueIds[r] = r+101;
        cudaStreamCreate(&(m_cudaStreams[r]));
        acc_set_cuda_stream(m_queueIds[r], m_cudaStreams[r]);
    }

    int rank;
    MPI_Comm_rank(communicator, &rank);
    for (const auto & queue : m_queueIds) {
        std::cout << "Rank " << rank << " " << queue.first << "/" << queue.second << std::endl;
    }
}

QueuesStreams::~QueuesStreams()
{
    for (int i = 0; i < m_cudaStreams.size(); ++i) {
        cudaStreamDestroy(m_cudaStreams[i]);
    }
}

cudaStream_t & QueuesStreams::getCudaStreamByRank(int rank)
{
    return m_cudaStreams.at(rank);
}
int & QueuesStreams::getOpenACCQueueByRank(int rank)
{
    return m_queueIds.at(rank);
}
