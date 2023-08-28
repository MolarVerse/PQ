#ifndef _TEST_INTRA_NON_BONDED_READER_HPP_

#define _TEST_INTRA_NON_BONDED_READER_HPP_

#include "engine.hpp"                 // for Engine
#include "intraNonBonded.hpp"         // for IntraNonBonded
#include "intraNonBondedReader.hpp"   // for IntraNonBondedReader
#include "molecule.hpp"               // for Molecule
#include "simulationBox.hpp"          // for SimulationBox

#include <gtest/gtest.h>   // for Test
#include <string>          // for allocator

/**
 * @class TestIntraNonBondedReader
 *
 * @brief Fixture class for testing the IntraNonBondedReader class
 *
 */
class TestIntraNonBondedReader : public ::testing::Test
{
  protected:
    engine::Engine                  *_engine;
    readInput::IntraNonBondedReader *_intraNonBondedReader;

    void SetUp() override
    {
        auto molecule1 = simulationBox::Molecule();
        molecule1.setNumberOfAtoms(3);
        molecule1.setMoltype(0);
        molecule1.setName("molecule1");

        _engine = new engine::Engine();

        _engine->getSimulationBox().addMoleculeType(molecule1);
        _engine->getIntraNonBonded().activate();

        _intraNonBondedReader = new readInput::IntraNonBondedReader("data/intraNonBondedReader/intraNonBonded.dat", *_engine);
    }

    void TearDown() override
    {
        delete _intraNonBondedReader;
        delete _engine;
    }
};

#endif   // _TEST_INTRA_NON_BONDED_READER_HPP_