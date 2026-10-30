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

#include <gtest/gtest.h>

#include <cstddef>

#include "engine.hpp"
#include "enums/potential.hpp"
#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "forceFieldSettings.hpp"
#include "moldescriptorReader.hpp"
#include "moleculeType.hpp"
#include "simulationBox.hpp"
#include "testMoldesctripotReader.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief tests constructor of input::molDescriptor::MoldescriptorReader
 *
 */
TEST_F(TestMoldescriptorReader, constructor)
{
    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptor.dat"
    );
    ASSERT_NO_THROW(
        input::molDescriptor::MoldescriptorReader reader(*_makeEngine(*_settings
        ))
    );
}

/**
 * @brief tests if line entry has at least 2 elements
 *
 */
TEST_F(TestMoldescriptorReader, argumentsInMoldescriptor)
{
    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptorWithOneWordLine.dat"
    );
    input::molDescriptor::MoldescriptorReader reader(*_makeEngine(*_settings));
    ASSERT_THROW_MSG(
        reader.read(),
        exc::MolDescriptorException,
        "Error in moldescriptor file at line 1"
    );
}

/**
 * @brief tests number of entries for molecule section in moldescriptor
 *
 */
TEST_F(TestMoldescriptorReader, argumentsInMoleculeSection)
{
    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptorWithErrorInAtomArguments.dat"
    );
    input::molDescriptor::MoldescriptorReader reader(*_makeEngine(*_settings));
    ASSERT_THROW_MSG(
        reader.read(),
        exc::MolDescriptorException,
        "Atom line in moldescriptor file at line 4 has to have 3 or 4 elements"
    );

    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptorWithErrorInAtomArguments2.dat"
    );
    input::molDescriptor::MoldescriptorReader reader2(*_makeEngine(*_settings));
    ASSERT_THROW_MSG(
        reader2.read(),
        exc::MolDescriptorException,
        "Atom line in moldescriptor file at line 5 has to have 3 or 4 elements"
    );

    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptorWithErrorInMolArguments.dat"
    );
    input::molDescriptor::MoldescriptorReader reader3(*_makeEngine(*_settings));
    ASSERT_THROW_MSG(
        reader3.read(),
        exc::MolDescriptorException,
        "Not enough arguments in moldescriptor file at line 3"
    );
}

/**
 * @brief test reading of moldescriptor.dat
 *
 */
TEST_F(TestMoldescriptorReader, moldescriptorReader)
{
    settings::FileSettings::setMolDescriptorFileName(
        "examples/setup/moldescriptor.dat"
    );
    input::molDescriptor::MoldescriptorReader reader(*_makeEngine(*_settings));
    ASSERT_NO_THROW(reader.read());
}

/**
 * @brief test reading of special types
 *
 */
TEST_F(TestMoldescriptorReader, specialTypes)
{
    auto engine = _makeEngine(*_settings);
    ASSERT_EQ(engine->getSimulationBox().getWaterType(), std::nullopt);
    settings::FileSettings::setMolDescriptorFileName(
        "examples/setup/moldescriptor.dat"
    );
    input::molDescriptor::readMolDescriptor(*engine);
    ASSERT_EQ(engine->getSimulationBox().getWaterType(), MolType{1});
    ASSERT_EQ(engine->getSimulationBox().getAmmoniaType(), MolType{2});
}

/**
 * @brief tests if there are to many atoms per moltype
 *
 */
TEST_F(TestMoldescriptorReader, toManyAtomsPerMoltype)
{
    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptorTooManyAtomsPerMoltype.dat"
    );
    input::molDescriptor::MoldescriptorReader reader2(*_makeEngine(*_settings));
    ASSERT_THROW_MSG(
        reader2.read(),
        exc::MolDescriptorException,
        "Error reading of moldescriptor stopped before last molecule was "
        "finished"
    );
}

/**
 * @brief tests if non coulombic force field is activated but no global can der
 * Waals parameter given
 *
 */
TEST_F(TestMoldescriptorReader, globalVdwTypes)
{
    _settings->forceFieldSettings.setType(ForceFieldType::ON);

    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptor_withGlobalVdwTypes.dat"
    );
    EXPECT_NO_THROW(
        input::molDescriptor::readMolDescriptor(*_makeEngine(*_settings))
    );

    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptor_withMissingGlobalVdwTypes.dat"
    );
    EXPECT_THROW_MSG(
        input::molDescriptor::readMolDescriptor(*_makeEngine(*_settings)),
        exc::MolDescriptorException,
        "Error in moldescriptor file at line 6 - force field noncoulombics is "
        "activated but no global van der Waals "
        "parameter given"
    );
}

/**
 * @brief tests the optional QM reference energy line of a molecule
 *
 */
TEST_F(TestMoldescriptorReader, qmReferenceEnergy)
{
    auto engine = _makeEngine(*_settings);
    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptorWithQMReferenceEnergy.dat"
    );
    input::molDescriptor::readMolDescriptor(*engine);

    auto &simBox = engine->getSimulationBox();
    EXPECT_DOUBLE_EQ(
        simBox.findMoleculeType(MolType{1}).getQMReferenceEnergy(),
        -2554.71
    );
    EXPECT_EQ(
        simBox.findMoleculeType(MolType{1}).getNumberOfAtoms(),
        size_t{3}
    );
    EXPECT_DOUBLE_EQ(
        simBox.findMoleculeType(MolType{2}).getQMReferenceEnergy(),
        0.0
    );

    settings::FileSettings::setMolDescriptorFileName(
        "data/moldescriptorReader/moldescriptorWithInvalidQMReferenceEnergy.dat"
    );
    EXPECT_THROW_MSG(
        input::molDescriptor::readMolDescriptor(*_makeEngine(*_settings)),
        exc::MolDescriptorException,
        "Invalid QM reference energy line in moldescriptor file at line 3 - "
        "expected \"qm_reference_energy <energy>\""
    );
}
