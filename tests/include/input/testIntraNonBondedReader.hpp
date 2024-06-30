/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _TEST_INTRA_NON_BONDED_READER_HPP_

#define _TEST_INTRA_NON_BONDED_READER_HPP_

#include <gtest/gtest.h>   // for Test

#include <string>   // for allocator

#include "fileSettings.hpp"           // for FileSettings
#include "intraNonBonded.hpp"         // for IntraNonBonded
#include "intraNonBondedReader.hpp"   // for IntraNonBondedReader
#include "mmmdEngine.hpp"             // for Engine
#include "moleculeType.hpp"           // for MoleculeType
#include "simulationBox.hpp"          // for SimulationBox

/**
 * @class TestIntraNonBondedReader
 *
 * @brief Fixture class for testing the IntraNonBondedReader class
 *
 */
class TestIntraNonBondedReader : public ::testing::Test
{
   protected:
    engine::Engine                                    *_engine;
    input::intraNonBondedReader::IntraNonBondedReader *_intraNonBondedReader;

    void SetUp() override
    {
        auto molecule1 = simulationBox::MoleculeType();
        molecule1.setNumberOfAtoms(3);
        molecule1.setMoltype(0);
        molecule1.setName("molecule1");

        // NOTE: use dummy engine for testing
        //       this is implemented by base class Engine
        //       and works therefore for all derived classes
        _engine = new engine::MMMDEngine();

        _engine->getSimulationBox().addMoleculeType(molecule1);
        _engine->getIntraNonBonded().activate();

        _intraNonBondedReader =
            new input::intraNonBondedReader::IntraNonBondedReader(
                "data/intraNonBondedReader/intraNonBonded.dat",
                *_engine
            );
        settings::FileSettings::setIsIntraNonBondedFileNameSet();
    }

    void TearDown() override
    {
        delete _intraNonBondedReader;
        delete _engine;
    }
};

#endif   // _TEST_INTRA_NON_BONDED_READER_HPP_