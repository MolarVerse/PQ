#ifndef _TEST_TOPOLOGY_SECTION_HPP_

#define _TEST_TOPOLOGY_SECTION_HPP_

#include "engine.hpp"          // for Engine
#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox

#include <gtest/gtest.h>   // for Test
#include <stdio.h>         // for remove
#include <string>          // for allocator, string

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
    std::string     _topologyFileName = "shake.top";

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
        ::remove(_topologyFileName.c_str());
    }
};

#endif   // _TEST_TOPOLOGY_SECTION_HPP_