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

#include <bitpit_containers.hpp>
#include <bitpit_communications.hpp>
#include <bitpit_patchkernel.hpp>

#include <mpi.h>
#include <vector>
#include <unordered_map>

class ExchangeBufferStreamer
{

public:
    ExchangeBufferStreamer(const size_t &itemSize);

    virtual ~ExchangeBufferStreamer() = default;

    size_t getItemSize() const;

    virtual void read(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) = 0;
    virtual void write(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) = 0;

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
    CudaStorageBufferStreamer(container_t *container, const size_t & readOffset, const size_t & writeOffset);
    CudaStorageBufferStreamer(container_t *container, const size_t & readOffset, const size_t & writeOffset, const size_t & itemSize);

    void read(const int &rank, bitpit::RecvBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;
    void write(const int &rank, bitpit::SendBuffer &buffer, const std::vector<long> &list = std::vector<long>()) override;

protected:
    size_t m_readOffset;
    size_t m_writeOffset;
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

class ListCommunicator : public bitpit::DataCommunicator
{

public:
    typedef std::vector<long> RankExchangeList;
    typedef std::unordered_map<int, RankExchangeList> ExchangeList;

    enum ListType {
        LIST_SEND,
        LIST_RECV
    };

    ListCommunicator(const MPI_Comm &communicator);

    virtual ~ListCommunicator() = default;

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
    GhostCommunicator(const bitpit::PatchKernel *patch);

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

#include "communications.tpp"

#endif

#endif
