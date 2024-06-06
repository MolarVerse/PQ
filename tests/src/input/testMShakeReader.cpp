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

#include "testMShakeReader.hpp"

#include <gtest/gtest.h>

#include <string>

#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "mShakeReader.hpp"
#include "mShakeReference.hpp"
#include "moleculeType.hpp"
#include "throwWithMessage.hpp"

TEST_F(TestMShakeReader, testConstructor)
{
    settings::FileSettings::setMShakeFileName("notExistingFile");
    const auto reader = input::mShake::MShakeReader(*_engine);
    EXPECT_EQ(reader.getFileName(), "notExistingFile");
}

TEST_F(TestMShakeReader, testProcessCommentLine)
{
    auto        reader          = input::mShake::MShakeReader(*_engine);
    std::string commentLine     = "# this is a comment line";
    auto        mShakeReference = constraints::MShakeReference();

    const auto error_message = std::format(
        "Unknown command in mShake file at line 0! The M-Shake file "
        "should be in the form a an extended xyz file. Here, the "
        "comment line should contain the molecule type from the "
        "moldescriptor file in the following form: 'MolType = 1;'. "
        "Please note that the syntax parsing works exactly like in the "
        "input file. Thus, it is case insensitive and the commands are "
        "separated by semicolons. Furthermore, the spaces around the "
        "'=' sign can be of arbitrary length (including also no spaces "
        "at all)."
    );

    EXPECT_THROW_MSG(
        reader.processCommentLine(commentLine, mShakeReference),
        customException::MShakeFileException,
        error_message
    );

    commentLine = "MolType = 1;";

    EXPECT_THROW_MSG(
        reader.processCommentLine(commentLine, mShakeReference),
        customException::MShakeFileException,
        "Molecule type 1 not found"
    );

    auto molType = simulationBox::MoleculeType(1);
    _engine->getSimulationBox().addMoleculeType(molType);

    reader.processCommentLine(commentLine, mShakeReference);
    EXPECT_EQ(mShakeReference.getMoleculeType().getMoltype(), 1);
}

TEST_F(TestMShakeReader, testProcessAtomLines)
{
    auto reader          = input::mShake::MShakeReader(*_engine);
    auto mShakeReference = constraints::MShakeReference();
    auto atomLine1       = "H 0.0 0.0 0.0";
    auto atomLine2       = "O 1.0 1.0 1.0  #asdfasdf";
    auto atomLine3       = "C 2.0 2.0 2.0 0.0";

    auto atomLines = std::vector<std::string>{atomLine1, atomLine2, atomLine3};

    const auto error_message = std::format(
        "Wrong number of elements in atom lines in mShake file "
        "starting at line 0! The M-Shake file should be in the form a "
        "an extended xyz file. Therefore, this line should contain the "
        "atom type and the coordinates of the atom."
    );

    EXPECT_THROW_MSG(
        reader.processAtomLines(atomLines, mShakeReference),
        customException::MShakeFileException,
        error_message
    );

    auto molType = simulationBox::MoleculeType(1);
    mShakeReference.setMoleculeType(molType);

    atomLines = std::vector<std::string>{atomLine1};

    EXPECT_THROW_MSG(
        reader.processAtomLines(atomLines, mShakeReference),
        customException::MShakeFileException,
        "Molecule type 1 has only one atom. M-Shake requires at least two "
        "atoms."
    );

    atomLine1 = "H 0.0 0.0 0.0";
    atomLine2 = "O 1.0 1.0 1.0  #asdfasdf";
    atomLine3 = "C 2.0 2.0 2.0   ";

    atomLines = std::vector<std::string>{atomLine1, atomLine2, atomLine3};

    EXPECT_THROW_MSG(
        reader.processAtomLines(atomLines, mShakeReference),
        customException::MShakeFileException,
        "Atom names in mShake file at line 0 do not match the atom names of "
        "the molecule type! The M-Shake file should be in the form a an "
        "extended xyz file. Therefore, the atom names in the atom lines should "
        "match the atom names of the molecule type from the restart file."
    );

    molType.addAtomName("H");
    molType.addAtomName("O");
    molType.addAtomName("C");

    mShakeReference.setMoleculeType(molType);

    reader.processAtomLines(atomLines, mShakeReference);

    auto atoms = mShakeReference.getAtoms();

    EXPECT_EQ(atoms.size(), 3);
    EXPECT_EQ(atoms[0].getName(), "H");
    EXPECT_EQ(atoms[1].getName(), "O");
    EXPECT_EQ(atoms[2].getName(), "C");
    EXPECT_EQ(atoms[0].getPosition()[0], 0.0);
    EXPECT_EQ(atoms[0].getPosition()[1], 0.0);
    EXPECT_EQ(atoms[0].getPosition()[2], 0.0);
    EXPECT_EQ(atoms[1].getPosition()[0], 1.0);
    EXPECT_EQ(atoms[1].getPosition()[1], 1.0);
    EXPECT_EQ(atoms[1].getPosition()[2], 1.0);
    EXPECT_EQ(atoms[2].getPosition()[0], 2.0);
    EXPECT_EQ(atoms[2].getPosition()[1], 2.0);
    EXPECT_EQ(atoms[2].getPosition()[2], 2.0);
}

TEST_F(TestMShakeReader, testReadMemberFunction)
{
    settings::FileSettings::setMShakeFileName("data/mshakeReader/mshake.dat");

    auto molType = simulationBox::MoleculeType(1);
    molType.addAtomName("H");
    molType.addAtomName("O");
    molType.addAtomName("C");
    _engine->getSimulationBox().addMoleculeType(molType);

    auto molType2 = simulationBox::MoleculeType(2);
    molType2.addAtomName("H");
    molType2.addAtomName("O");
    _engine->getSimulationBox().addMoleculeType(molType2);

    input::mShake::readMShake(*_engine);

    auto mShakeReferences = _engine->getConstraints().getMShakeReferences();

    EXPECT_EQ(mShakeReferences.size(), 2);
    EXPECT_EQ(mShakeReferences[0].getMoleculeType().getMoltype(), 1);
    EXPECT_EQ(mShakeReferences[1].getMoleculeType().getMoltype(), 2);
    EXPECT_EQ(mShakeReferences[0].getAtoms().size(), 3);
    EXPECT_EQ(mShakeReferences[1].getAtoms().size(), 2);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[0].getName(), "H");
    EXPECT_EQ(mShakeReferences[0].getAtoms()[1].getName(), "O");
    EXPECT_EQ(mShakeReferences[0].getAtoms()[2].getName(), "C");
    EXPECT_EQ(mShakeReferences[1].getAtoms()[0].getName(), "H");
    EXPECT_EQ(mShakeReferences[1].getAtoms()[1].getName(), "O");
    EXPECT_EQ(mShakeReferences[0].getAtoms()[0].getPosition()[0], 0.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[0].getPosition()[1], 0.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[0].getPosition()[2], 0.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[1].getPosition()[0], 1.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[1].getPosition()[1], 1.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[1].getPosition()[2], 1.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[2].getPosition()[0], 2.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[2].getPosition()[1], 2.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[2].getPosition()[2], 2.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[0].getPosition()[0], 0.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[0].getPosition()[1], 0.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[0].getPosition()[2], 0.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[1].getPosition()[0], 1.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[1].getPosition()[1], 1.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[1].getPosition()[2], 1.0);
}

TEST_F(TestMShakeReader, testRead)
{
    settings::FileSettings::setMShakeFileName("data/mshakeReader/mshake.dat");
    auto reader = input::mShake::MShakeReader(*_engine);

    auto molType = simulationBox::MoleculeType(1);
    molType.addAtomName("H");
    molType.addAtomName("O");
    molType.addAtomName("C");
    _engine->getSimulationBox().addMoleculeType(molType);

    auto molType2 = simulationBox::MoleculeType(2);
    molType2.addAtomName("H");
    molType2.addAtomName("O");
    _engine->getSimulationBox().addMoleculeType(molType2);

    reader.read();

    auto mShakeReferences = _engine->getConstraints().getMShakeReferences();

    EXPECT_EQ(mShakeReferences.size(), 2);
    EXPECT_EQ(mShakeReferences[0].getMoleculeType().getMoltype(), 1);
    EXPECT_EQ(mShakeReferences[1].getMoleculeType().getMoltype(), 2);
    EXPECT_EQ(mShakeReferences[0].getAtoms().size(), 3);
    EXPECT_EQ(mShakeReferences[1].getAtoms().size(), 2);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[0].getName(), "H");
    EXPECT_EQ(mShakeReferences[0].getAtoms()[1].getName(), "O");
    EXPECT_EQ(mShakeReferences[0].getAtoms()[2].getName(), "C");
    EXPECT_EQ(mShakeReferences[1].getAtoms()[0].getName(), "H");
    EXPECT_EQ(mShakeReferences[1].getAtoms()[1].getName(), "O");
    EXPECT_EQ(mShakeReferences[0].getAtoms()[0].getPosition()[0], 0.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[0].getPosition()[1], 0.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[0].getPosition()[2], 0.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[1].getPosition()[0], 1.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[1].getPosition()[1], 1.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[1].getPosition()[2], 1.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[2].getPosition()[0], 2.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[2].getPosition()[1], 2.0);
    EXPECT_EQ(mShakeReferences[0].getAtoms()[2].getPosition()[2], 2.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[0].getPosition()[0], 0.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[0].getPosition()[1], 0.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[0].getPosition()[2], 0.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[1].getPosition()[0], 1.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[1].getPosition()[1], 1.0);
    EXPECT_EQ(mShakeReferences[1].getAtoms()[1].getPosition()[2], 1.0);
}
