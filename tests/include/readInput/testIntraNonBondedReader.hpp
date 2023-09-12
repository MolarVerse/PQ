#ifndef _TEST_INTRA_NON_BONDED_READER_HPP_

#define _TEST_INTRA_NON_BONDED_READER_HPP_

#include "engine.hpp"                 // for Engine
#include "fileSettings.hpp"           // for FileSettings
#include "intraNonBonded.hpp"         // for IntraNonBonded
#include "intraNonBondedReader.hpp"   // for IntraNonBondedReader
#include "moleculeType.hpp"           // for MoleculeType
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
    engine::Engine                                  *_engine;
    readInput::intraNonBonded::IntraNonBondedReader *_intraNonBondedReader;

    void SetUp() override
    {
        auto molecule1 = simulationBox::MoleculeType();
        molecule1.setNumberOfAtoms(3);
        molecule1.setMoltype(0);
        molecule1.setName("molecule1");

        _engine = new engine::Engine();

        _engine->getSimulationBox().addMoleculeType(molecule1);
        _engine->getIntraNonBonded().activate();

        _intraNonBondedReader =
            new readInput::intraNonBonded::IntraNonBondedReader("data/intraNonBondedReader/intraNonBonded.dat", *_engine);
        settings::FileSettings::setIsIntraNonBondedFileNameSet();
    }

    void TearDown() override
    {
        delete _intraNonBondedReader;
        delete _engine;
    }
};

#endif   // _TEST_INTRA_NON_BONDED_READER_HPP_