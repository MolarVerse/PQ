#ifndef _TEST_TOPOLOGY_READER_HPP_

#define _TEST_TOPOLOGY_READER_HPP_

#include "engine.hpp"
#include "topologyReader.hpp"

#include <gtest/gtest.h>
#include <string>

class TestTopologyReader : public ::testing::Test
{
  protected:
    engine::Engine        *_engine;
    setup::TopologyReader *_topologyReader;

    void SetUp() override
    {
        _engine         = new engine::Engine();
        _topologyReader = new setup::TopologyReader("data/topologyReader/topology.top", *_engine);
    }

    void TearDown() override
    {
        delete _topologyReader;
        delete _engine;
    }
};

#endif   // _TEST_TOPOLOGY_READER_HPP_