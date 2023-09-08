#ifndef _TEST_TOPOLOGY_READER_HPP_

#define _TEST_TOPOLOGY_READER_HPP_

#include "engine.hpp"           // for Engine
#include "fileSettings.hpp"     // for FileSettings
#include "molecule.hpp"         // for Molecule
#include "simulationBox.hpp"    // for SimulationBox
#include "topologyReader.hpp"   // for TopologyReader

#include <gtest/gtest.h>   // for Test
#include <string>          // for allocator

/**
 * @class TestTopologyReader
 *
 * @brief Fixture class for testing the TopologyReader class
 *
 */
class TestTopologyReader : public ::testing::Test
{
  protected:
    engine::Engine                      *_engine;
    readInput::topology::TopologyReader *_topologyReader;

    void SetUp() override
    {
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(1);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(2);

        _engine = new engine::Engine();

        _engine->getSimulationBox().addMolecule(molecule1);
        _engine->getSimulationBox().addMolecule(molecule2);

        _topologyReader = new readInput::topology::TopologyReader("data/topologyReader/topology.top", *_engine);
        settings::FileSettings::setIsTopologyFileNameSet();
    }

    void TearDown() override
    {
        delete _topologyReader;
        delete _engine;
    }
};

#endif   // _TEST_TOPOLOGY_READER_HPP_