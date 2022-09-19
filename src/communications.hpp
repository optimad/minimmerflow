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

#if ENABLE_MPI

#ifndef __MINIMMERFLOW_COMMUNICATIONS_HPP__
#define __MINIMMERFLOW_COMMUNICATIONS_HPP__

#include <containers.hpp>

#include <bitpit_containers.hpp>
#include <bitpit_communications.hpp>
#include <bitpit_patchkernel.hpp>

#include <mpi.h>
#include <vector>
#include <unordered_map>

#include "cuda_runtime_api.h"

//class ExchangeRecvBuffer : public bitpit::RecvBuffer
//{
//public:
//    using bitpit::RecvBuffer();
//    using bitpit::getFront();
//}
//class ExchangeSendBuffer : public bitpit::SendBuffer
//{
//public:
//    using bitpit::SendBuffer();
//    using bitpit::getFront();
//}

class ExchangeBufferStreamer
{

public:
    ExchangeBufferStreamer(const size_t &itemSize);

    virtual ~ExchangeBufferStreamer() = default;

    size_t getItemSize() const;

    virtual void read(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) = 0;
    virtual void finalizeRead(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>());
    virtual void write(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) = 0;
    virtual void finalizeWrite(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>());

private:
    size_t m_itemSize;

    void setItemSize(const size_t &itemSize);

};

template<typename container_t>
class BaseListBufferStreamer : public ExchangeBufferStreamer
{

public:
    typedef container_t container_type;

    BaseListBufferStreamer(container_t *container);
    BaseListBufferStreamer(container_t *container, const size_t &itemSize);

    container_t & getContainer();

protected:
    container_t *m_container;

};

template<typename container_t>
class ListBufferStreamer : public BaseListBufferStreamer<container_t>
{

public:
    using BaseListBufferStreamer<container_t>::BaseListBufferStreamer;

    void read(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void write(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;

};

template<typename container_t>
class CudaStorageBufferStreamer : public BaseListBufferStreamer<container_t>
{

public:
    //using BaseListBufferStreamer<container_t>::BaseListBufferStreamer;
    //CudaStorageBufferStreamer(container_t *container, const size_t & readOffset, const size_t & writeOffset);
    CudaStorageBufferStreamer(container_t *container, const size_t & readOffset, const size_t & writeOffset, const size_t & itemSize, const std::unordered_map<int, size_t> & rankOffsets, int fieldId);
    ~CudaStorageBufferStreamer();

    void read(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void write(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void finalizeWrite(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;

    void initializeCUDAObjects();
    std::unordered_map<int, cudaStream_t> m_cudaStreams; //like this rank cannot share GPU

protected:
    size_t m_readOffset;
    size_t m_writeOffset;
    std::unordered_map<int, size_t> m_rankOffsets;
};

namespace cuda_streamer {
void scatter(double * buffer, double ** container, std::size_t listSize, std::size_t * list);
}

template<typename container_t>
class CudaStorageCollectionBufferStreamer : public BaseListBufferStreamer<container_t>
{

public:
    //using BaseListBufferStreamer<container_t>::BaseListBufferStreamer;
    //CudaStorageBufferStreamer(container_t *container, const size_t & readOffset, const size_t & writeOffset);
    CudaStorageCollectionBufferStreamer(container_t *container, const size_t & readOffset, const size_t & writeOffset, const size_t & itemSize);
    ~CudaStorageCollectionBufferStreamer();

    void read(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void finalizeRead(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void write(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void finalizeWrite(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;

    void initializeCUDAObjects();
    void initializePointers(ScalarPiercedStorageCollection<double> * storage, std::unordered_map<int, ScalarStorage<std::size_t>> * targets);
    std::unordered_map<int, cudaStream_t> m_cudaStreams; //like this rank cannot share GPU

protected:
    size_t m_readOffset;
    size_t m_writeOffset;

    ScalarPiercedStorageCollection<double> * m_deviceStorage;
    std::unordered_map<int, ScalarStorage<std::size_t>> * m_targetLists;
    std::unordered_map<int, ScalarStorage<std::size_t>> * m_sourceLists;
};


template<typename value_t>
class PiercedStorageBufferStreamer : public ListBufferStreamer<bitpit::PiercedStorage<value_t, long>>
{

public:
    PiercedStorageBufferStreamer(bitpit::PiercedStorage<value_t, long> *container);
    PiercedStorageBufferStreamer(bitpit::PiercedStorage<value_t, long> *container, const size_t &itemSize);

    void read(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void write(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;

};

template<typename value_t>
class PiercedStorageCollectionBufferStreamer : public ListBufferStreamer<PiercedStorageCollection<value_t, value_t>>
{

public:
    typedef value_t value_type;

    PiercedStorageCollectionBufferStreamer(PiercedStorageCollection<value_t, value_t> *container);
    PiercedStorageCollectionBufferStreamer(PiercedStorageCollection<value_t, value_t> *container, const size_t &itemSize);

    void read(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void write(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;

};

#if ENABLE_MPI
template<typename value_t>
using ValuePiercedStorageBufferStreamer = PiercedStorageBufferStreamer<value_t>;
template<typename value_t>
using ValuePiercedStorageCollectionBufferStreamer = PiercedStorageCollectionBufferStreamer<value_t>;
#endif



class ListCommunicator : public bitpit::DataCommunicator
{

public:
    typedef std::vector<long> RankExchangeList;
    typedef std::unordered_map<int, RankExchangeList> ExchangeList;

    enum ListType {
        LIST_SEND,
        LIST_RECV
    };

    ListCommunicator(const MPI_Comm &communicator, std::string name);

    ~ListCommunicator();
//    virtual ~ListCommunicator() = default;

    size_t getItemSize() const;

    const ExchangeList & getSendList() const;
    const RankExchangeList & getSendList(int rank) const;
    const ExchangeList & getRecvList() const;
    const RankExchangeList & getRecvList(int rank) const;
    virtual void setExchangeLists(const ExchangeList &sendList, const ExchangeList &recvList);
    virtual void setExchangeList(ListType listcontainer_type, const ExchangeList &list);

    bool hasData() const;
    void addData(ExchangeBufferStreamer *streamer);
    void addData(ExchangeBufferStreamer *writer, ExchangeBufferStreamer *reader);

    void startAllExchanges();
    void completeAllExchanges();

    void completeAllRecvs();
    int completeAnyRecv(const std::vector<int> &blacklist = std::vector<int>());

    void completeAllSends();

    void remapExchangeLists(const std::unordered_map<long, long> &mapper);
    void remapExchangeLists(const std::unordered_map<int, std::vector<long>> &mapper);

    void remapSendList(const std::unordered_map<long, long> &mapper);
    void remapSendList(const std::unordered_map<int, std::vector<long>> &mapper);

    void remapRecvList(const std::unordered_map<long, long> &mapper);
    void remapRecvList(const std::unordered_map<int, std::vector<long>> &mapper);

    void initializeCudaObjects();
    void finalizeCudaObjects();

    std::string m_name;

protected:
    size_t m_itemSize;
    ExchangeList m_sendList;
    ExchangeList m_recvList;

    void updateExchangeInfo();

    ExchangeList scatterExchangeList(const ExchangeList &inputList);

    void remapList(ExchangeList &list, const std::unordered_map<long, long> &mapper);
    void remapList(ExchangeList &list, const std::unordered_map<int, std::vector<long>> &mapper);

    virtual const RankExchangeList & getStreamableSendList(int rank, ExchangeBufferStreamer *reader);
    virtual const RankExchangeList & getStreamableRecvList(int rank, ExchangeBufferStreamer *writer);

private:
    std::vector<ExchangeBufferStreamer *> m_writers;
    std::vector<ExchangeBufferStreamer *> m_readers;

};

class GhostCommunicator : public ListCommunicator
{

public:
    GhostCommunicator(const bitpit::PatchKernel *patch, std::string name);

    void resetExchangeLists();
    void setExchangeLists(const ExchangeList &sendList, const ExchangeList &recvList);
    void setExchangeList(ListType listcontainer_type, const ExchangeList &list);

protected:
    void createStreamableLists();

    const RankExchangeList & getStreamableSendList(int rank, ExchangeBufferStreamer *reader);
    const RankExchangeList & getStreamableRecvList(int rank, ExchangeBufferStreamer *writer);

private:
    const bitpit::PatchKernel *m_patch;

    ExchangeList m_sendListIds;
    ExchangeList m_recvListIds;

    ExchangeList sequentialIndexesConversion(const ExchangeList &list);

};

class CommunicationsManager {

public:
    static MPI_Comm getCommunicator();
    static void setCommunicator(MPI_Comm communicator);

    static int getMasterRank();
    static bool isMaster();

private:
    static CommunicationsManager & instance();

    MPI_Comm m_communicator;

    CommunicationsManager();
    CommunicationsManager(const CommunicationsManager &other) = delete;
    CommunicationsManager(CommunicationsManager &&other) = delete;

};

class OpenACCStreams
{
public:
    OpenACCStreams(int nFields);
    ~OpenACCStreams();
    std::vector<int> m_streamIds;

private:
    std::vector<cudaStream_t> m_cudaStreams;

};

#include "communications.tpp"

#endif

#endif
