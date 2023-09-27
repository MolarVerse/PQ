/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for MolDescriptorException
#include "fileSettings.hpp"              // for FileSettings
#include "forceFieldClass.hpp"           // for ForceField
#include "moldescriptorReader.hpp"       // for MoldescriptorReader
#include "simulationBox.hpp"             // for SimulationBox
#include "testMoldesctripotReader.hpp"   // for TestMoldescriptorReader
#include "throwWithMessage.hpp"          // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult, testing
#include <gtest/gtest.h>   // for TestInfo (ptr only), TEST_F
#include <iosfwd>          // for std
#include <string>          // for allocator, basic_string

using namespace std;
using namespace ::testing;
using namespace readInput::molDescriptor;
using namespace customException;

/**
 * @brief tests constructor of MoldescriptorReader
 *
 */
TEST_F(TestMoldescriptorReader, constructor)
{
    settings::FileSettings::setMolDescriptorFileName("data/moldescriptorReader/moldescriptor.dat");
    ASSERT_NO_THROW(MoldescriptorReader reader(*_engine));
}

/**
 * @brief tests if line entry has at least 2 elements
 *
 */
TEST_F(TestMoldescriptorReader, argumentsInMoldescriptor)
{
    settings::FileSettings::setMolDescriptorFileName("data/moldescriptorReader/moldescriptorWithOneWordLine.dat");
    MoldescriptorReader reader(*_engine);
    ASSERT_THROW_MSG(reader.read(), MolDescriptorException, "Error in moldescriptor file at line 1");
}

/**
 * @brief tests number of entries for molecule section in moldescriptor
 *
 */
TEST_F(TestMoldescriptorReader, argumentsInMoleculeSection)
{
    settings::FileSettings::setMolDescriptorFileName("data/moldescriptorReader/moldescriptorWithErrorInAtomArguments.dat");
    MoldescriptorReader reader(*_engine);
    ASSERT_THROW_MSG(
        reader.read(), MolDescriptorException, "Atom line in moldescriptor file at line 4 has to have 3 or 4 elements");

    settings::FileSettings::setMolDescriptorFileName("data/moldescriptorReader/moldescriptorWithErrorInAtomArguments2.dat");
    MoldescriptorReader reader2(*_engine);
    ASSERT_THROW_MSG(
        reader2.read(), MolDescriptorException, "Atom line in moldescriptor file at line 5 has to have 3 or 4 elements");

    settings::FileSettings::setMolDescriptorFileName("data/moldescriptorReader/moldescriptorWithErrorInMolArguments.dat");
    MoldescriptorReader reader3(*_engine);
    ASSERT_THROW_MSG(reader3.read(), MolDescriptorException, "Not enough arguments in moldescriptor file at line 3");
}

/**
 * @brief test reading of moldescriptor.dat
 *
 */
TEST_F(TestMoldescriptorReader, moldescriptorReader)
{
    settings::FileSettings::setMolDescriptorFileName("examples/setup/moldescriptor.dat");
    MoldescriptorReader reader(*_engine);
    ASSERT_NO_THROW(reader.read());
}

/**
 * @brief test reading of special types
 *
 */
TEST_F(TestMoldescriptorReader, specialTypes)
{
    settings::FileSettings::setMolDescriptorFileName("examples/setup/moldescriptor.dat");
    readMolDescriptor(*_engine);
    ASSERT_EQ(_engine->getSimulationBox().getWaterType(), 1);
    ASSERT_EQ(_engine->getSimulationBox().getAmmoniaType(), 2);
}

/**
 * @brief tests if there are to many atoms per moltype
 *
 */
TEST_F(TestMoldescriptorReader, toManyAtomsPerMoltype)
{
    settings::FileSettings::setMolDescriptorFileName("data/moldescriptorReader/moldescriptorTooManyAtomsPerMoltype.dat");
    MoldescriptorReader reader2(*_engine);
    ASSERT_THROW_MSG(
        reader2.read(), MolDescriptorException, "Error reading of moldescriptor stopped before last molecule was finished");
}

/**
 * @brief tests if non coulombic force field is activated but no global can der Waals parameter given
 *
 */
TEST_F(TestMoldescriptorReader, globalVdwTypes)
{
    _engine->getForceFieldPtr()->activateNonCoulombic();

    settings::FileSettings::setMolDescriptorFileName("data/moldescriptorReader/moldescriptor_withGlobalVdwTypes.dat");
    EXPECT_NO_THROW(readMolDescriptor(*_engine));

    settings::FileSettings::setMolDescriptorFileName("data/moldescriptorReader/moldescriptor_withMissingGlobalVdwTypes.dat");
    EXPECT_THROW_MSG(readMolDescriptor(*_engine),
                     MolDescriptorException,
                     "Error in moldescriptor file at line 6 - force field noncoulombics is activated but no global van der Waals "
                     "parameter given");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}