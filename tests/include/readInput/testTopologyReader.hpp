#ifndef _TEST_TOPOLOGY_READER_HPP_

#define _TEST_TOPOLOGY_READER_HPP_

#include "engine.hpp"
#include "topologyReader.hpp"

#include <gtest/gtest.h>
#include <string>

class TestTopologyReader : public ::testing::Test
{
  protected:
    engine::Engine            *_engine;
    readInput::TopologyReader *_topologyReader;

    void SetUp() override
    {
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(1);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(2);

        _engine = new engine::Engine();

        _engine->getSimulationBox().addMolecule(molecule1);
        _engine->getSimulationBox().addMolecule(molecule2);

        _topologyReader = new readInput::TopologyReader("data/topologyReader/topology.top", *_engine);
    }

    void TearDown() override
    {
        delete _topologyReader;
        delete _engine;
    }
};

#endif   // _TEST_TOPOLOGY_READER_HPP_