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

#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ

#include <cstddef>    // for size_t
#include <fstream>    // for ifstream, std
#include <iterator>   // for std::ranges::distance
#include <memory>     // for shared_ptr, __shared_ptr_access
#include <ranges>
#include <string>   // for string, stod, allocator, basic_string
#include <vector>   // for vector

#include "atom.hpp"                 // for Atom
#include "atomSection.hpp"          // for AtomSection
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for RstFileException, customException
#include "gmock/gmock.h"            // for ElementsAre, MakePredicateFormatter
#include "gtest/gtest.h"            // for Message, TestPartResult
#include "molecule.hpp"             // for Molecule
#include "moleculeType.hpp"         // for MoleculeType
#include "restartFileSection.hpp"   // for RstFileSection, AtomSection
#include "simulationBox.hpp"        // for SimulationBox
#include "testRestartFileSection.hpp"   // for TestAtomSection
#include "throwWithMessage.hpp"         // for ASSERT_THROW_MSG

using namespace input::restartFile;

/**
 * @brief tests the keyword function
 *
 */
TEST_F(TestAtomSection, keyword) { EXPECT_EQ(_section->keyword(), ""); }

/**
 * @brief tests the isHeader function
 *
 */
TEST_F(TestAtomSection, isHeader) { EXPECT_FALSE(_section->isHeader()); }

/**
 * @brief tests the numberOfArguments function
 *
 */
TEST_F(TestAtomSection, numberOfArguments)
{
    _section->_lineNumber = 7;
    for (size_t i = 0; i < 25; ++i)
        if (i % 3 != 0 || i < 6 || i > 21)
        {
            auto line = std::vector<std::string>(i);
            ASSERT_THROW_MSG(
                _section->process(line, *_engine),
                customException::RstFileException,
                "Error in line 7: Atom section must have 6, 9, 12, 15, 18 or "
                "21 elements"
            );
        }
}

/**
 * @brief tests if moltype is not found in process function
 *
 */
TEST_F(TestAtomSection, moltypeNotFound)
{
    auto line = std::vector<std::string>(21);
    line[2]   = "1";
    ASSERT_THROW_MSG(
        _section->process(line, *_engine),
        customException::RstFileException,
        "Molecule type 1 not found"
    );
}

TEST_F(TestAtomSection, notEnoughElementsInLine)
{
    _section->_lineNumber = 7;
    auto line             = std::vector<std::string>(21);
    line[2]               = "1";
    for (size_t i = 3; i < 21; ++i) line[i] = "1.0";

    std::string filename = "data/atomSection/testNotEnoughAtomsInMolecule.rst";

    auto molecule = simulationBox::MoleculeType(1);
    molecule.setNumberOfAtoms(3);
    _engine->getSimulationBox().getMoleculeTypes().push_back(molecule);

    std::ifstream fp(filename);
    _section->_fp = &fp;

    ASSERT_THROW(
        _section->process(line, *_engine),
        customException::RstFileException
    );

    line[2] = "1";

    std::string filename2 =
        "data/atomSection/testNotEnoughAtomsInMolecule2.rst";
    std::ifstream fp2(filename2);
    _section->_fp = &fp2;

    ASSERT_THROW(
        _section->process(line, *_engine),
        customException::RstFileException
    );
}

TEST_F(TestAtomSection, numberOfArgumentsWithinMolecule)
{
    auto line = std::vector<std::string>(21);
    line[2]   = "1";
    for (size_t i = 3; i < 21; ++i) line[i] = "1.0";

    std::string filename =
        "data/atomSection/testNumberOfArgumentsWithinMolecule.rst";

    auto molecule = simulationBox::MoleculeType(1);
    molecule.setNumberOfAtoms(3);
    _engine->getSimulationBox().getMoleculeTypes().push_back(molecule);

    std::ifstream fp(filename);
    _section->_fp = &fp;

    ASSERT_THROW(
        _section->process(line, *_engine),
        customException::RstFileException
    );
}

TEST_F(TestAtomSection, testProcess)
{
    auto line = std::vector<std::string>(21);
    line[2]   = "1";
    for (size_t i = 3; i < 21; ++i) line[i] = "1.0";

    std::string filename = "data/atomSection/testProcess.rst";

    auto molecule = simulationBox::MoleculeType(1);
    molecule.setNumberOfAtoms(3);
    _engine->getSimulationBox().addMoleculeType(molecule);

    auto molecule2 = simulationBox::MoleculeType(2);
    molecule2.setNumberOfAtoms(4);
    _engine->getSimulationBox().addMoleculeType(molecule2);

    std::ifstream fp(filename);
    _section->_fp = &fp;

    _section->process(line, *_engine);

    line    = std::vector<std::string>(21);
    line[2] = "2";
    for (size_t i = 3; i < 21; ++i) line[i] = "1.0";

    _section->process(line, *_engine);

    line    = std::vector<std::string>(21);
    line[2] = "1";
    for (size_t i = 3; i < 21; ++i) line[i] = "1.0";

    _section->process(line, *_engine);

    EXPECT_EQ(_engine->getSimulationBox().getMolecules().size(), 3);

    EXPECT_EQ(_engine->getSimulationBox().getMolecules()[0].getMoltype(), 1);
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getNumberOfAtoms(),
        3
    );

    EXPECT_EQ(_engine->getSimulationBox().getMolecules()[1].getMoltype(), 2);
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[1].getNumberOfAtoms(),
        4
    );

    EXPECT_EQ(_engine->getSimulationBox().getMolecules()[2].getMoltype(), 1);
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[2].getNumberOfAtoms(),
        3
    );

    EXPECT_EQ(_engine->getSimulationBox().getNumberOfQMAtoms(), 0);
}

TEST_F(TestAtomSection, testProcessAtomLine)
{
    simulationBox::Molecule molecule(1);

    auto line = std::vector<std::string>(21);
    line[0]   = "Ar";
    for (size_t i = 3; i < 21; ++i) line[i] = std::to_string(i + i / 10.0);

    dynamic_cast<AtomSection *>(_section)
        ->processAtomLine(line, _engine->getSimulationBox(), molecule);

    ASSERT_THAT(
        molecule.getAtomPosition(0),
        testing::ElementsAre(stod(line[3]), stod(line[4]), stod(line[5]))
    );
    ASSERT_THAT(
        molecule.getAtomVelocity(0),
        testing::ElementsAre(stod(line[6]), stod(line[7]), stod(line[8]))
    );
    ASSERT_THAT(
        molecule.getAtomForce(0),
        testing::ElementsAre(stod(line[9]), stod(line[10]), stod(line[11]))
    );

    ASSERT_EQ(molecule.getAtom(0).getAtomTypeName(), line[0]);
}

TEST_F(TestAtomSection, testProcessQMAtomLine)
{
    auto line = std::vector<std::string>(21);
    line[0]   = "Ar";
    for (size_t i = 3; i < 21; ++i) line[i] = std::to_string(i + i / 10.0);

    dynamic_cast<AtomSection *>(_section)->processQMAtomLine(
        line,
        _engine->getSimulationBox()
    );

    settings::Settings::setJobtype(settings::JobType::QM_MD);
    auto atoms      = _engine->getSimulationBox().getQMAtomsNew();
    auto first_atom = *atoms.begin();

    ASSERT_EQ(_engine->getSimulationBox().getNumberOfQMAtoms(), 1);
    ASSERT_THAT(
        first_atom->getPosition(),
        testing::ElementsAre(stod(line[3]), stod(line[4]), stod(line[5]))
    );
    ASSERT_THAT(
        first_atom->getVelocity(),
        testing::ElementsAre(stod(line[6]), stod(line[7]), stod(line[8]))
    );
    ASSERT_THAT(
        first_atom->getForce(),
        testing::ElementsAre(stod(line[9]), stod(line[10]), stod(line[11]))
    );

    ASSERT_EQ(first_atom->getAtomTypeName(), line[0]);

    settings::Settings::setJobtype(settings::JobType::NONE);
}