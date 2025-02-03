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

#include <gtest/gtest.h>   // for CmpHelperFloatingPointEQ

#include <cmath>    // for cbrt
#include <memory>   // for make_shared, __shared_ptr_access
#include <string>   // for basic_string
#include <vector>   // for vector

#include "atom.hpp"                          // for Atom
#include "constants/conversionFactors.hpp"   // for _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"     // for MolDescriptorException, InputFileException
#include "fileSettings.hpp"   // for FileSettings
#include "forceFieldSettings.hpp"      // for ForceFieldSettings
#include "gtest/gtest.h"               // for Message, TestPartResult
#include "molecule.hpp"                // for Molecule
#include "moleculeType.hpp"            // for MoleculeType
#include "potentialSettings.hpp"       // for PotentialSettings
#include "simulationBox.hpp"           // for SimulationBox
#include "simulationBoxSettings.hpp"   // for SimulationBoxSettings
#include "simulationBoxSetup.hpp"   // for SimulationBoxSetup, setupSimulationBox
#include "testSetup.hpp"            // for TestSetup
#include "vector3d.hpp"             // for Vec3D

using setup::simulationBox::SimulationBoxSetup;

TEST_F(TestSetup, setAtomNames)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    const ::simulationBox::Molecule qmMolecule(0);

    ::simulationBox::MoleculeType moleculeType(1);
    moleculeType.setNumberOfAtoms(3);
    moleculeType.addAtomName("zN");
    moleculeType.addAtomName("H");
    moleculeType.addAtomName("H");

    _engine->getSimulationBox().addMolecule(molecule);
    _engine->getSimulationBox().addMolecule(qmMolecule);
    _engine->getSimulationBox().addMoleculeType(moleculeType);

    _engine->getSimulationBox().addAtom(atom1);
    _engine->getSimulationBox().addAtom(atom2);
    _engine->getSimulationBox().addAtom(atom3);

    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomNames();

    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomName(0),
        "Zn"
    );
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomName(1),
        "H"
    );
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomName(2),
        "H"
    );
}

TEST_F(TestSetup, setAtomTypes)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    const ::simulationBox::Molecule qmMolecule(0);

    ::simulationBox::MoleculeType moleculeType(1);
    moleculeType.setNumberOfAtoms(3);
    moleculeType.addAtomType(0);
    moleculeType.addAtomType(1);
    moleculeType.addAtomType(2);
    moleculeType.addExternalAtomType(0);
    moleculeType.addExternalAtomType(1);
    moleculeType.addExternalAtomType(2);

    _engine->getSimulationBox().addMolecule(molecule);
    _engine->getSimulationBox().addMolecule(qmMolecule);
    _engine->getSimulationBox().addMoleculeType(moleculeType);

    _engine->getSimulationBox().addAtom(atom1);
    _engine->getSimulationBox().addAtom(atom2);
    _engine->getSimulationBox().addAtom(atom3);

    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomTypes();

    EXPECT_EQ(_engine->getSimulationBox().getMolecules()[0].getAtomType(0), 0);
    EXPECT_EQ(_engine->getSimulationBox().getMolecules()[0].getAtomType(1), 1);
    EXPECT_EQ(_engine->getSimulationBox().getMolecules()[0].getAtomType(2), 2);
    EXPECT_EQ(
        _engine->getSimulationBox()
            .getMolecules()[0]
            .getAtom(0)
            .getExternalAtomType(),
        0
    );
    EXPECT_EQ(
        _engine->getSimulationBox()
            .getMolecules()[0]
            .getAtom(1)
            .getExternalAtomType(),
        1
    );
    EXPECT_EQ(
        _engine->getSimulationBox()
            .getMolecules()[0]
            .getAtom(2)
            .getExternalAtomType(),
        2
    );
}

TEST_F(TestSetup, setExternalVDWTypes)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    const ::simulationBox::Molecule qmMolecule(0);

    ::simulationBox::MoleculeType moleculeType(1);
    moleculeType.setNumberOfAtoms(3);
    moleculeType.addExternalGlobalVDWType(0);
    moleculeType.addExternalGlobalVDWType(1);
    moleculeType.addExternalGlobalVDWType(2);

    _engine->getSimulationBox().addMolecule(molecule);
    _engine->getSimulationBox().addMolecule(qmMolecule);
    _engine->getSimulationBox().addMoleculeType(moleculeType);

    _engine->getSimulationBox().addAtom(atom1);
    _engine->getSimulationBox().addAtom(atom2);
    _engine->getSimulationBox().addAtom(atom3);

    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setExternalVDWTypes();

    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getExternalGlobalVDWTypes(
        )[0],
        0
    );
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getExternalGlobalVDWTypes(
        )[1],
        1
    );
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getExternalGlobalVDWTypes(
        )[2],
        2
    );
}

TEST_F(TestSetup, setPartialCharges)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    const ::simulationBox::Molecule qmMolecule(0);

    ::simulationBox::MoleculeType moleculeType(1);
    moleculeType.setNumberOfAtoms(3);
    moleculeType.addPartialCharge(0.0);
    moleculeType.addPartialCharge(1.0);
    moleculeType.addPartialCharge(2.0);

    _engine->getSimulationBox().addMolecule(molecule);
    _engine->getSimulationBox().addMolecule(qmMolecule);
    _engine->getSimulationBox().addMoleculeType(moleculeType);

    _engine->getSimulationBox().addAtom(atom1);
    _engine->getSimulationBox().addAtom(atom2);
    _engine->getSimulationBox().addAtom(atom3);

    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setPartialCharges();

    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getPartialCharges()[0],
        0.0
    );
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getPartialCharges()[1],
        1.0
    );
    EXPECT_EQ(
        _engine->getSimulationBox().getMolecules()[0].getPartialCharges()[2],
        2.0
    );
}

TEST_F(TestSetup, testSetAtomMasses)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    _engine->getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomMasses();

    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomMass(0),
        12.0107
    );
    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomMass(1),
        1.00794
    );
    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomMass(2),
        15.9994
    );
}

TEST_F(TestSetup, testSetAtomMassesThrowsError)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("L");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    _engine->getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    ASSERT_THROW(
        simulationBoxSetup.setAtomMasses(),
        customException::MolDescriptorException
    );
}

TEST_F(TestSetup, testSetAtomicNumbers)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    _engine->getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomicNumbers();

    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomicNumber(0),
        6
    );
    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomicNumber(1),
        1
    );
    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getMolecules()[0].getAtomicNumber(2),
        8
    );
}

TEST_F(TestSetup, testSetAtomicNumbersThrowsError)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("L");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    _engine->getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    ASSERT_THROW(
        simulationBoxSetup.setAtomicNumbers(),
        customException::MolDescriptorException
    );
}

TEST_F(TestSetup, testSetMolMass)
{
    ::simulationBox::Molecule molecule1(1);
    molecule1.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule1.addAtom(atom1);
    molecule1.addAtom(atom2);
    molecule1.addAtom(atom3);

    ::simulationBox::Molecule molecule2(2);
    molecule2.setNumberOfAtoms(2);
    const auto atom4 = std::make_shared<::simulationBox::Atom>();
    const auto atom5 = std::make_shared<::simulationBox::Atom>();
    atom4->setName("H");
    atom5->setName("H");
    molecule2.addAtom(atom4);
    molecule2.addAtom(atom5);

    _engine->getSimulationBox().getMolecules().push_back(molecule1);
    _engine->getSimulationBox().getMolecules().push_back(molecule2);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomMasses();
    simulationBoxSetup.calculateMolMasses();

    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getMolecules()[0].getMolMass(),
        12.0107 + 1 * 1.00794 + 15.9994
    );
}

TEST_F(TestSetup, testSetTotalCharge)
{
    ::simulationBox::Molecule molecule(1);
    molecule.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    molecule.setPartialCharges({0.1, 0.2, -0.4});

    _engine->getSimulationBox().getMolecules().push_back(molecule);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.calculateTotalCharge();

    EXPECT_DOUBLE_EQ(_engine->getSimulationBox().getTotalCharge(), -0.1);
}

TEST_F(TestSetup, noDensityNoBox)
{
    settings::SimulationBoxSettings::setDensitySet(false);
    settings::SimulationBoxSettings::setBoxSet(false);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    ASSERT_THROW(
        simulationBoxSetup.checkBoxSettings(),
        customException::UserInputException
    );
}

TEST_F(TestSetup, noDensity)
{
    _engine->getSimulationBox().setTotalMass(6000);
    _engine->getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    settings::SimulationBoxSettings::setDensitySet(false);
    settings::SimulationBoxSettings::setBoxSet(true);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine->getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getDensity(),
        constants::_AMU_PER_ANGSTROM3_TO_KG_PER_L_
    );
}

TEST_F(TestSetup, testNoBox)
{
    _engine->getSimulationBox().setTotalMass(6000);
    _engine->getSimulationBox().setDensity(
        constants::_AMU_PER_ANGSTROM3_TO_KG_PER_L_
    );
    settings::SimulationBoxSettings::setBoxSet(false);
    settings::SimulationBoxSettings::setDensitySet(true);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine->getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getBoxDimensions()[2],
        cbrt(6000.0)
    );
}

TEST_F(TestSetup, testBoxAndDensitySet)
{
    _engine->getSimulationBox().setTotalMass(6000);
    _engine->getSimulationBox().setDensity(12341243.1234
    );   // this should be ignored
    _engine->getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    settings::SimulationBoxSettings::setDensitySet(true);
    settings::SimulationBoxSettings::setBoxSet(true);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(_engine->getSimulationBox().getVolume(), 6000.0);
    EXPECT_DOUBLE_EQ(
        _engine->getSimulationBox().getDensity(),
        constants::_AMU_PER_ANGSTROM3_TO_KG_PER_L_
    );
}

TEST_F(TestSetup, testCheckRcCutoff)
{
    _engine->getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    settings::PotentialSettings::setCoulombRadiusCutOff(14.0);
    SimulationBoxSetup simulationBoxSetup(*_engine);
    EXPECT_THROW(
        simulationBoxSetup.checkRcCutoff(),
        customException::InputFileException
    );

    SimulationBoxSetup simulationBox2Setup(*_engine);
    settings::PotentialSettings::setCoulombRadiusCutOff(4.0);
    EXPECT_NO_THROW(simulationBox2Setup.checkRcCutoff());
}

TEST_F(TestSetup, testCheckZeroVelocities)
{
    SimulationBoxSetup simBoxSetup(*_engine);
    auto              &simBox = _engine->getSimulationBox();

    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    atom1->setVelocity({8e-10, 0.0, 0.0});
    atom2->setVelocity({1.0, 2.0, 3.0});
    simBox.getAtoms().push_back(atom1);
    simBox.getAtoms().push_back(atom2);
    simBoxSetup.checkZeroVelocities();

    EXPECT_FALSE(SimulationBoxSetup::getZeroVelocities());

    atom2->setVelocity({0.0, 0.0, 0.0});
    simBoxSetup.checkZeroVelocities();

    EXPECT_FALSE(SimulationBoxSetup::getZeroVelocities());

    atom1->setVelocity({0.0, 0.0, 0.0});
    simBoxSetup.checkZeroVelocities();

    EXPECT_TRUE(SimulationBoxSetup::getZeroVelocities());
}

/**
 * @brief testing full setup of simulation box
 *
 * @TODO: this test is not complete, it only tests the functions that are called
 * in the setup
 *
 */
TEST_F(TestSetup, testFullSetup)
{
    settings::ForceFieldSettings::activate();

    ::simulationBox::Molecule molecule1(1);
    molecule1.setNumberOfAtoms(3);
    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    const auto atom3 = std::make_shared<::simulationBox::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule1.addAtom(atom1);
    molecule1.addAtom(atom2);
    molecule1.addAtom(atom3);

    molecule1.setPartialCharges({0.1, 0.2, -0.4});

    ::simulationBox::Molecule molecule2(2);
    molecule2.setNumberOfAtoms(2);
    const auto atom4 = std::make_shared<::simulationBox::Atom>();
    const auto atom5 = std::make_shared<::simulationBox::Atom>();
    atom4->setName("H");
    atom5->setName("H");
    molecule2.addAtom(atom4);
    molecule2.addAtom(atom5);

    molecule2.setPartialCharges({0.1, 0.2});

    _engine->getSimulationBox().getMolecules().push_back(molecule1);
    _engine->getSimulationBox().getMolecules().push_back(molecule2);

    auto moleculeType1 = ::simulationBox::MoleculeType(1);
    auto moleculeType2 = ::simulationBox::MoleculeType(2);

    moleculeType1.setNumberOfAtoms(3);
    moleculeType1.addAtomName("C");
    moleculeType1.addAtomName("H");
    moleculeType1.addAtomName("O");
    moleculeType1.addPartialCharge(0.1);
    moleculeType1.addPartialCharge(0.2);
    moleculeType1.addPartialCharge(-0.4);
    moleculeType1.addAtomType(0);
    moleculeType1.addAtomType(0);
    moleculeType1.addAtomType(0);
    moleculeType1.addExternalAtomType(0);
    moleculeType1.addExternalAtomType(0);
    moleculeType1.addExternalAtomType(0);
    moleculeType1.addExternalGlobalVDWType(0);
    moleculeType1.addExternalGlobalVDWType(1);
    moleculeType1.addExternalGlobalVDWType(2);

    moleculeType2.setNumberOfAtoms(2);
    moleculeType2.addAtomName("H");
    moleculeType2.addAtomName("H");
    moleculeType2.addPartialCharge(0.1);
    moleculeType2.addPartialCharge(0.2);
    moleculeType2.addAtomType(0);
    moleculeType2.addAtomType(0);
    moleculeType2.addExternalAtomType(0);
    moleculeType2.addExternalAtomType(0);
    moleculeType2.addExternalGlobalVDWType(0);
    moleculeType2.addExternalGlobalVDWType(1);

    _engine->getSimulationBox().addMoleculeType(moleculeType1);
    _engine->getSimulationBox().addMoleculeType(moleculeType2);

    _engine->getSimulationBox().setTotalMass(33.0);
    _engine->getSimulationBox().setDensity(12341243.1234
    );   // this should be ignored
    _engine->getSimulationBox().setBoxDimensions({10.0, 20.0, 30.0});
    settings::PotentialSettings::setCoulombRadiusCutOff(4.0);

    EXPECT_NO_THROW(setup::simulationBox::setupSimulationBox(*_engine));
}

TEST_F(TestSetup, testWriteSetupInfo)
{
    _engine->getEngineOutput().getLogOutput().setFilename("default.log");
    auto &simBox = _engine->getSimulationBox();

    const auto atom1 = std::make_shared<::simulationBox::Atom>();
    const auto atom2 = std::make_shared<::simulationBox::Atom>();
    simBox.getAtoms().push_back(atom1);
    simBox.getAtoms().push_back(atom2);
    auto simBox2 = simBox;
    simBox.copy(simBox2);

    simBox.setTotalMass(13.05);
    simBox.setTotalCharge(-7.03);
    simBox.setDegreesOfFreedom(4);
    simBox.setDensity(12341243.1234);
    simBox.setVolume(758373.194);
    simBox.setBoxDimensions({5.73, 9.93, 11.14});
    settings::PotentialSettings::setCoulombRadiusCutOff(5.79);
    settings::FileSettings::setStartFileName("input.rst");

    SimulationBoxSetup(*_engine).writeSetupInfo();
    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         number of atoms:          2");
    getline(file, line);
    EXPECT_EQ(line, "         total mass:              13.05000 g/mol");
    getline(file, line);
    EXPECT_EQ(line, "         total charge:            -7.03000");
    getline(file, line);
    EXPECT_EQ(line, "         unconstrained DOF:        4");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         density:         12341243.12340 kg/L");
    getline(file, line);
    EXPECT_EQ(line, "         volume:            758373.19400 \u212b³");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    // clang-format off
    EXPECT_EQ(line,"         box dimensions:         5.73000 \u212b        9.93000 \u212b       11.14000 \u212b");
    getline(file, line);
    EXPECT_EQ(line,"         box angles:            90.00000°        90.00000°        90.00000°");
    // clang-format on
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         coulomb cutoff:         5.79000 \u212b");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         velocities taken from start file \"input.rst\"");

    ::remove("default.log");
}

TEST_F(TestSetup, testWriteSetupInfoMaxwellTrueNonZeroVelocities)
{
    _engine->getEngineOutput().getLogOutput().setFilename("default.log");

    settings::SimulationBoxSettings::setInitializeVelocities(
        settings::InitVelocities::TRUE
    );
    SimulationBoxSetup::setZeroVelocities(false);
    settings::FileSettings::setStartFileName("input.rst");

    SimulationBoxSetup(*_engine).writeSetupInfo();
    std::ifstream file("default.log");
    std::string   line;
    for (int i = 0; i < 13; i++)
    {
        getline(file, line);
    }
    getline(file, line);
    EXPECT_EQ(
        line,
        "WARNING: Ignoring 'init_velocities' because non-zero velocities in "
        "\"input.rst\""
    );
    getline(file, line);
    EXPECT_EQ(line, "         velocities taken from start file \"input.rst\"");

    ::remove("default.log");
}

TEST_F(TestSetup, testWriteSetupInfoMaxwellTrueZeroVelocities)
{
    _engine->getEngineOutput().getLogOutput().setFilename("default.log");

    settings::SimulationBoxSettings::setInitializeVelocities(
        settings::InitVelocities::TRUE
    );
    SimulationBoxSetup::setZeroVelocities(true);

    SimulationBoxSetup(*_engine).writeSetupInfo();
    std::ifstream file("default.log");
    std::string   line;
    for (int i = 0; i < 13; i++)
    {
        getline(file, line);
    }
    getline(file, line);
    EXPECT_EQ(
        line,
        "         velocities initialized with Maxwell-Boltzmann distribution"
    );

    ::remove("default.log");
}

TEST_F(TestSetup, testWriteSetupInfoMaxwellForceNonZeroVelocities)
{
    _engine->getEngineOutput().getLogOutput().setFilename("default.log");

    settings::SimulationBoxSettings::setInitializeVelocities(
        settings::InitVelocities::FORCE
    );
    SimulationBoxSetup::setZeroVelocities(false);

    SimulationBoxSetup(*_engine).writeSetupInfo();
    std::ifstream file("default.log");
    std::string   line;
    for (int i = 0; i < 13; i++)
    {
        getline(file, line);
    }
    getline(file, line);
    EXPECT_EQ(
        line,
        "         velocities initialized with Maxwell-Boltzmann distribution"
    );

    ::remove("default.log");
}