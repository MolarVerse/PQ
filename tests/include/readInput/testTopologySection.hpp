#ifndef _TEST_TOPOLOGY_SECTION_HPP_

#define _TEST_TOPOLOGY_SECTION_HPP_

#include "engine.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <string>

/**
 * @class TestTopologySection
 *
 * @brief Fixture class for testing the TopologySection class
 *
 */
class TestTopologySection : public ::testing::Test
{
  protected:
    engine::Engine *_engine;
    std::string     _topologyFilename = "shake.top";

    void SetUp() override
    {
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(1);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(3);

        _engine = new engine::Engine();

        _engine->getSimulationBox().addMolecule(molecule1);
        _engine->getSimulationBox().addMolecule(molecule2);
    }

    void TearDown() override
    {
        delete _engine;
        ::remove(_topologyFilename.c_str());
    }
};

#endif   // _TEST_TOPOLOGY_SECTION_HPP_