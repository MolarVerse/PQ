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

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "atom.hpp"
#include "engine.hpp"
#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "forceFieldSettings.hpp"
#include "generalSettings.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "orthorhombicBox.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "simulationBoxSettings.hpp"
#include "simulationBoxSetup.hpp"
#include "strongTypes.hpp"
#include "testSetup.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "unit_types.hpp"
#include "units.hpp"

TEST_F(TestSetup, setAtomNames)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    const ::molsys::Molecule qmMolecule{MolType{0}};

    ::molsys::MoleculeType moleculeType(MolType{1});
    moleculeType.setNumberOfAtoms(3);
    moleculeType.addAtomName("zN");
    moleculeType.addAtomName("H");
    moleculeType.addAtomName("H");

    _engine->get_simulationBox().addMolecule(molecule);
    _engine->get_simulationBox().addMolecule(qmMolecule);
    _engine->get_simulationBox().addMoleculeType(moleculeType);

    _engine->get_simulationBox().addAtom(atom1);
    _engine->get_simulationBox().addAtom(atom2);
    _engine->get_simulationBox().addAtom(atom3);

    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomNames();

    const auto molceule = _engine->get_simulationBox().getMolecules()[0];
    EXPECT_EQ(molceule.getAtomName(AtomIndex{0}), "Zn");
    EXPECT_EQ(molceule.getAtomName(AtomIndex{1}), "H");
    EXPECT_EQ(molceule.getAtomName(AtomIndex{2}), "H");
}

TEST_F(TestSetup, setAtomTypes)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    const ::molsys::Molecule qmMolecule{MolType{0}};

    ::molsys::MoleculeType moleculeType(MolType{1});
    moleculeType.setNumberOfAtoms(3);
    moleculeType.addAtomType(AtomType{0});
    moleculeType.addAtomType(AtomType{1});
    moleculeType.addAtomType(AtomType{2});
    moleculeType.addExternalAtomType(ExtAtomType{0});
    moleculeType.addExternalAtomType(ExtAtomType{1});
    moleculeType.addExternalAtomType(ExtAtomType{2});

    _engine->get_simulationBox().addMolecule(molecule);
    _engine->get_simulationBox().addMolecule(qmMolecule);
    _engine->get_simulationBox().addMoleculeType(moleculeType);

    _engine->get_simulationBox().addAtom(atom1);
    _engine->get_simulationBox().addAtom(atom2);
    _engine->get_simulationBox().addAtom(atom3);

    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomTypes();

    auto molecule2 = _engine->get_simulationBox().getMolecules()[0];
    EXPECT_EQ(molecule2.getAtomType(AtomIndex{0}), AtomType{0});
    EXPECT_EQ(molecule2.getAtomType(AtomIndex{1}), AtomType{1});
    EXPECT_EQ(molecule2.getAtomType(AtomIndex{2}), AtomType{2});
    EXPECT_EQ(
        molecule2.getAtom(AtomIndex{0}).getExternalAtomType(),
        ExtAtomType{0}
    );
    EXPECT_EQ(
        molecule2.getAtom(AtomIndex{1}).getExternalAtomType(),
        ExtAtomType{1}
    );
    EXPECT_EQ(
        molecule2.getAtom(AtomIndex{2}).getExternalAtomType(),
        ExtAtomType{2}
    );
}

TEST_F(TestSetup, setExternalVDWTypes)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    const ::molsys::Molecule qmMolecule{MolType{0}};

    ::molsys::MoleculeType moleculeType(MolType{1});
    moleculeType.setNumberOfAtoms(3);
    moleculeType.addExternalGlobalVDWType(ExtVdwType{0});
    moleculeType.addExternalGlobalVDWType(ExtVdwType{1});
    moleculeType.addExternalGlobalVDWType(ExtVdwType{2});

    _engine->get_simulationBox().addMolecule(molecule);
    _engine->get_simulationBox().addMolecule(qmMolecule);
    _engine->get_simulationBox().addMoleculeType(moleculeType);

    _engine->get_simulationBox().addAtom(atom1);
    _engine->get_simulationBox().addAtom(atom2);
    _engine->get_simulationBox().addAtom(atom3);

    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setExternalVDWTypes();

    const auto &moleculeResult = _engine->get_simulationBox().getMolecules()[0];

    EXPECT_EQ(moleculeResult.getExternalGlobalVDWTypes()[0], ExtVdwType{0});
    EXPECT_EQ(moleculeResult.getExternalGlobalVDWTypes()[1], ExtVdwType{1});
    EXPECT_EQ(moleculeResult.getExternalGlobalVDWTypes()[2], ExtVdwType{2});
}

TEST_F(TestSetup, setPartialCharges)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    const ::molsys::Molecule qmMolecule{MolType{0}};

    ::molsys::MoleculeType moleculeType(MolType{1});
    moleculeType.setNumberOfAtoms(3);
    moleculeType.addPartialCharge(0.0 * elementary_charge);
    moleculeType.addPartialCharge(1.0 * elementary_charge);
    moleculeType.addPartialCharge(2.0 * elementary_charge);

    _engine->get_simulationBox().addMolecule(molecule);
    _engine->get_simulationBox().addMolecule(qmMolecule);
    _engine->get_simulationBox().addMoleculeType(moleculeType);

    _engine->get_simulationBox().addAtom(atom1);
    _engine->get_simulationBox().addAtom(atom2);
    _engine->get_simulationBox().addAtom(atom3);

    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setPartialCharges();

    const auto &moleculeResult = _engine->get_simulationBox().getMolecules()[0];

    EXPECT_EQ(moleculeResult.getPartialCharges()[0], 0.0 * elementary_charge);
    EXPECT_EQ(moleculeResult.getPartialCharges()[1], 1.0 * elementary_charge);
    EXPECT_EQ(moleculeResult.getPartialCharges()[2], 2.0 * elementary_charge);
}

TEST_F(TestSetup, testSetAtomMasses)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    _engine->get_simulationBox().getMolecules().push_back(molecule);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomMasses();

    const auto &moleculeResult = _engine->get_simulationBox().getMolecules()[0];

    EXPECT_DOUBLE_EQ(
        moleculeResult.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol),
        12.0107
    );
    EXPECT_DOUBLE_EQ(
        moleculeResult.getAtomMass(AtomIndex{1}).numerical_value_in(g_per_mol),
        1.00794
    );
    EXPECT_DOUBLE_EQ(
        moleculeResult.getAtomMass(AtomIndex{2}).numerical_value_in(g_per_mol),
        15.9994
    );
}

TEST_F(TestSetup, testSetAtomMassesThrowsError)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("L");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    _engine->get_simulationBox().getMolecules().push_back(molecule);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    ASSERT_THROW_MSG(
        simulationBoxSetup.setAtomMasses(),
        exc::MolDescriptorException,
        "Invalid atom name \"l\""
    );
}

TEST_F(TestSetup, testSetAtomicNumbers)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    _engine->get_simulationBox().getMolecules().push_back(molecule);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomicNumbers();

    const auto &moleculeResult = _engine->get_simulationBox().getMolecules()[0];

    EXPECT_EQ(moleculeResult.getAtomicNumber(AtomIndex{0}), AtomNumber{6});
    EXPECT_EQ(moleculeResult.getAtomicNumber(AtomIndex{1}), AtomNumber{1});
    EXPECT_EQ(moleculeResult.getAtomicNumber(AtomIndex{2}), AtomNumber{8});
}

TEST_F(TestSetup, testSetAtomicNumbersThrowsError)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("L");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    _engine->get_simulationBox().getMolecules().push_back(molecule);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    ASSERT_THROW_MSG(
        simulationBoxSetup.setAtomicNumbers(),
        exc::MolDescriptorException,
        "Invalid atom name \"l\""
    );
}

TEST_F(TestSetup, testSetMolMass)
{
    ::molsys::Molecule molecule1{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule1.addAtom(atom1);
    molecule1.addAtom(atom2);
    molecule1.addAtom(atom3);

    ::molsys::Molecule molecule2{MolType{2}};
    const auto         atom4 = std::make_shared<::molsys::Atom>();
    const auto         atom5 = std::make_shared<::molsys::Atom>();
    atom4->setName("H");
    atom5->setName("H");
    molecule2.addAtom(atom4);
    molecule2.addAtom(atom5);

    _engine->get_simulationBox().getMolecules().push_back(molecule1);
    _engine->get_simulationBox().getMolecules().push_back(molecule2);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.setAtomMasses();

    EXPECT_DOUBLE_EQ(
        _engine->get_simulationBox()
            .getMolecules()[0]
            .getMolMass()
            .numerical_value_in(g_per_mol),
        12.0107 + (1 * 1.00794) + 15.9994
    );
}

TEST_F(TestSetup, testSetTotalCharge)
{
    ::molsys::Molecule molecule{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    molecule.setPartialCharges(
        {0.1 * elementary_charge,
         0.2 * elementary_charge,
         -0.4 * elementary_charge}
    );

    _engine->get_simulationBox().getMolecules().push_back(molecule);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.calculateTotalCharge();

    EXPECT_DOUBLE_EQ(
        _engine->get_simulationBox().getTotalCharge().numerical_value_in(
            elementary_charge
        ),
        -0.1
    );
}

TEST_F(TestSetup, noDensityNoBox)
{
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    ASSERT_THROW_MSG(
        simulationBoxSetup.checkBoxSettings(),
        exc::UserInputException,
        "Box dimensions and density not set"
    );
}

TEST_F(TestSetup, noDensity)
{
    _engine->get_simulationBox().setTotalMass(6000 * g_per_mol);

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 20.0 * angstrom, 30.0 * angstrom}
    };
    _engine->get_simulationBox().setBox(orthoBox, true);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(
        _engine->get_simulationBox().getVolume().numerical_value_in(
            angstrom * angstrom * angstrom
        ),
        6000.0
    );
    EXPECT_DOUBLE_EQ(
        _engine->get_simulationBox().getDensity().numerical_value_in(
            kg_per_litre
        ),
        (1.0 * g_per_mol / angstrom3 / avogadro_constant)
            .numerical_value_in(kg_per_litre)
    );
}

TEST_F(TestSetup, testNoBox)
{
    _engine->get_simulationBox().setTotalMass(6000 * g_per_mol);
    _settings->simulationBoxSettings.setDensityByUser(
        1.0 * g_per_mol / angstrom3 / avogadro_constant
    );
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(
        _engine->get_simulationBox().getVolume().numerical_value_in(
            angstrom * angstrom * angstrom
        ),
        6000.0
    );
    EXPECT_DOUBLE_EQ(
        _engine->get_simulationBox().getBoxDimensions()[2].numerical_value_in(
            angstrom
        ),
        cbrt(6000.0)
    );
}

TEST_F(TestSetup, testBoxAndDensitySet)
{
    _engine->get_simulationBox().setTotalMass(6000 * g_per_mol);
    _engine->get_simulationBox().setDensity(
        12341243.1234 * kg_per_litre
    );   // this should be ignored

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 20.0 * angstrom, 30.0 * angstrom}
    };
    _engine->get_simulationBox().setBox(orthoBox, true);
    _settings->simulationBoxSettings.setDensityByUser(0.0 * kg_per_litre);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    simulationBoxSetup.checkBoxSettings();

    EXPECT_DOUBLE_EQ(
        _engine->get_simulationBox().getVolume().numerical_value_in(
            angstrom * angstrom * angstrom
        ),
        6000.0
    );
    EXPECT_DOUBLE_EQ(
        _engine->get_simulationBox().getDensity().numerical_value_in(
            kg_per_litre
        ),
        (1.0 * g_per_mol / angstrom3 / avogadro_constant)
            .numerical_value_in(kg_per_litre)
    );
}

TEST_F(TestSetup, testCheckRcCutoff)
{
    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 20.0 * angstrom, 30.0 * angstrom}
    };
    _engine->get_simulationBox().setBox(orthoBox, false);
    _settings->potentialSettings.setCoulombRadiusCutOffByUser(14.0 * angstrom);
    setup::SimulationBoxSetup simulationBoxSetup(*_engine);
    EXPECT_THROW_MSG(
        simulationBoxSetup.checkRcCutoff(),
        exc::InputFileException,
        "Rc cutoff is larger than half of the minimal box dimension of 10 Å."
    );

    setup::SimulationBoxSetup simulationBox2Setup(*_engine);
    _settings->potentialSettings.setCoulombRadiusCutOffByUser(4.0 * angstrom);
    EXPECT_NO_THROW(simulationBox2Setup.checkRcCutoff());
}

TEST_F(TestSetup, testCheckZeroVelocities)
{
    setup::SimulationBoxSetup simBoxSetup(*_engine);
    auto                     &simBox = _engine->get_simulationBox();

    const auto atom1 = std::make_shared<::molsys::Atom>();
    const auto atom2 = std::make_shared<::molsys::Atom>();
    atom1->setVelocity(linalg::Vec3D{8e-10, 0.0, 0.0} * angstrom / s);
    atom2->setVelocity(linalg::Vec3D{1.0, 2.0, 3.0} * angstrom / s);
    simBox.getAtoms().push_back(atom1);
    simBox.getAtoms().push_back(atom2);
    simBoxSetup.checkZeroVelocities();

    EXPECT_FALSE(setup::SimulationBoxSetup::getZeroVelocities());

    atom2->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);
    simBoxSetup.checkZeroVelocities();

    EXPECT_FALSE(setup::SimulationBoxSetup::getZeroVelocities());

    atom1->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);
    simBoxSetup.checkZeroVelocities();

    EXPECT_TRUE(setup::SimulationBoxSetup::getZeroVelocities());
}

/**
 * @brief testing full setup of simulation box
 *
 * TODO: this test is not complete, it only tests the functions that are called
 * in the setup
 *
 */
TEST_F(TestSetup, testFullSetup)
{
    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::ON);

    ::molsys::Molecule molecule1{MolType{1}};
    const auto         atom1 = std::make_shared<::molsys::Atom>();
    const auto         atom2 = std::make_shared<::molsys::Atom>();
    const auto         atom3 = std::make_shared<::molsys::Atom>();
    atom1->setName("C");
    atom2->setName("H");
    atom3->setName("O");
    molecule1.addAtom(atom1);
    molecule1.addAtom(atom2);
    molecule1.addAtom(atom3);

    molecule1.setPartialCharges(
        {0.1 * elementary_charge,
         0.2 * elementary_charge,
         -0.4 * elementary_charge}
    );

    ::molsys::Molecule molecule2{MolType{2}};
    const auto         atom4 = std::make_shared<::molsys::Atom>();
    const auto         atom5 = std::make_shared<::molsys::Atom>();
    atom4->setName("H");
    atom5->setName("H");
    molecule2.addAtom(atom4);
    molecule2.addAtom(atom5);

    molecule2.setPartialCharges(
        {0.1 * elementary_charge, 0.2 * elementary_charge}
    );

    _engine->get_simulationBox().getMolecules().push_back(molecule1);
    _engine->get_simulationBox().getMolecules().push_back(molecule2);

    auto moleculeType1 = molsys::MoleculeType(MolType{1});
    auto moleculeType2 = molsys::MoleculeType(MolType{2});

    moleculeType1.setNumberOfAtoms(3);
    moleculeType1.addAtomName("C");
    moleculeType1.addAtomName("H");
    moleculeType1.addAtomName("O");
    moleculeType1.addPartialCharge(0.1 * elementary_charge);
    moleculeType1.addPartialCharge(0.2 * elementary_charge);
    moleculeType1.addPartialCharge(-0.4 * elementary_charge);
    moleculeType1.addAtomType(AtomType{0});
    moleculeType1.addAtomType(AtomType{0});
    moleculeType1.addAtomType(AtomType{0});
    moleculeType1.addExternalAtomType(ExtAtomType{0});
    moleculeType1.addExternalAtomType(ExtAtomType{0});
    moleculeType1.addExternalAtomType(ExtAtomType{0});
    moleculeType1.addExternalGlobalVDWType(ExtVdwType{0});
    moleculeType1.addExternalGlobalVDWType(ExtVdwType{1});
    moleculeType1.addExternalGlobalVDWType(ExtVdwType{2});

    moleculeType2.setNumberOfAtoms(2);
    moleculeType2.addAtomName("H");
    moleculeType2.addAtomName("H");
    moleculeType2.addPartialCharge(0.1 * elementary_charge);
    moleculeType2.addPartialCharge(0.2 * elementary_charge);
    moleculeType2.addAtomType(AtomType{0});
    moleculeType2.addAtomType(AtomType{0});
    moleculeType2.addExternalAtomType(ExtAtomType{0});
    moleculeType2.addExternalAtomType(ExtAtomType{0});
    moleculeType2.addExternalGlobalVDWType(ExtVdwType{0});
    moleculeType2.addExternalGlobalVDWType(ExtVdwType{1});

    _engine->get_simulationBox().addMoleculeType(moleculeType1);
    _engine->get_simulationBox().addMoleculeType(moleculeType2);

    _engine->get_simulationBox().setTotalMass(33.0 * g_per_mol);
    _engine->get_simulationBox().setDensity(
        12341243.1234 * kg_per_litre
    );   // this should be ignored

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 20.0 * angstrom, 30.0 * angstrom}
    };
    _settings->fileSettings.setStartFileNameByUser("input.rst");
    _engine->get_simulationBox().setBox(orthoBox, true);
    _settings->potentialSettings.setCoulombRadiusCutOffByUser(4.0 * angstrom);

    EXPECT_NO_THROW(setup::setupSimulationBox(*_engine));
}

TEST_F(TestSetup, testWriteSetupInfo)
{
    _engine->getEngineOutput().getLogOutput().setFilename("default.log", false);
    auto &simBox = _engine->get_simulationBox();

    const auto atom1 = std::make_shared<::molsys::Atom>();
    const auto atom2 = std::make_shared<::molsys::Atom>();
    simBox.getAtoms().push_back(atom1);
    simBox.getAtoms().push_back(atom2);
    auto simBox2 = simBox;
    simBox.copy(simBox2);

    simBox.setTotalMass(13.05 * g_per_mol);
    simBox.setTotalCharge(-7.03 * elementary_charge);
    simBox.setDegreesOfFreedom(4);
    simBox.setDensity(12341243.1234 * kg_per_litre);
    molsys::OrthorhombicBox orthoBox{
        {5.73 * angstrom, 9.93 * angstrom, 11.14 * angstrom}
    };
    simBox.setBox(orthoBox, false);
    _settings->potentialSettings.setCoulombRadiusCutOffByUser(5.79 * angstrom);
    _settings->fileSettings.setStartFileNameByUser("input.rst");
    setup::SimulationBoxSetup::setZeroVelocities(false);

    setup::SimulationBoxSetup(*_engine).writeSetupInfo();
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
    EXPECT_EQ(line, "         volume:               633.85375 \u212b³");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(
        line,
        "         box dimensions:         5.73000 \u212b        9.93000 \u212b "
        "      11.14000 \u212b"
    );
    getline(file, line);
    EXPECT_EQ(
        line,
        "         box angles:            90.00000°        90.00000°        "
        "90.00000°"
    );
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         coulomb cutoff:         5.79000 \u212b");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         velocities taken from start file \"input.rst\"");

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
}

TEST_F(TestSetup, testWriteSetupInfoMaxwellTrueNonZeroVelocities)
{
    _engine->getEngineOutput().getLogOutput().setFilename("default.log", false);
    _settings->simulationBoxSettings.setInitializeVelocsByUser(
        InitVelocities::TRUE
    );
    setup::SimulationBoxSetup::setZeroVelocities(false);
    _settings->fileSettings.setStartFileNameByUser("input.rst");

    setup::SimulationBoxSetup(*_engine).writeSetupInfo();
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

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
}

TEST_F(TestSetup, testWriteSetupInfoMaxwellTrueZeroVelocities)
{
    _engine->getEngineOutput().getLogOutput().setFilename("default.log", false);

    _settings->simulationBoxSettings.setInitializeVelocsByUser(
        InitVelocities::TRUE
    );
    setup::SimulationBoxSetup::setZeroVelocities(true);

    setup::SimulationBoxSetup(*_engine).writeSetupInfo();
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

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
}

TEST_F(TestSetup, testWriteSetupInfoMaxwellForceNonZeroVelocities)
{
    _engine->getEngineOutput().getLogOutput().setFilename("default.log", false);

    _settings->simulationBoxSettings.setInitializeVelocsByUser(
        InitVelocities::FORCE
    );
    setup::SimulationBoxSetup::setZeroVelocities(false);

    setup::SimulationBoxSetup(*_engine).writeSetupInfo();
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

    const auto errorCode = std::remove("default.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: default.log";
}

class TestZeroVelocityWarning : public TestSetup
{
   protected:
    void        SetUp() override;
    std::string _writeSetupInfoAndReadLog();
};

void TestZeroVelocityWarning::SetUp()
{
    TestSetup::SetUp();
    _settings->kinetics.setStartTemperatureByUser(toTemperature(300.0));
}

std::string TestZeroVelocityWarning::_writeSetupInfoAndReadLog()
{
    _engine->getEngineOutput().getLogOutput().setFilename("zeroVel.log", false);
    _settings->fileSettings.setStartFileNameByUser("input.rst");

    setup::SimulationBoxSetup(*_engine).writeSetupInfo();

    std::ifstream     file("zeroVel.log");
    std::stringstream content;
    content << file.rdbuf();
    file.close();

    const auto errorCode = std::remove("zeroVel.log");
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: zeroVel.log";

    return content.str();
}

static constexpr auto ZERO_VELOCITY_WARNING =
    "WARNING: All velocities in \"input.rst\" are zero: the MD starts at 0 K. "
    "Set 'init_velocities = true' to draw them from a Maxwell-Boltzmann "
    "distribution; 'start_temp' only starts the temperature ramp";

TEST_F(TestZeroVelocityWarning, AllZeroInitOff)
{
    setup::SimulationBoxSetup::setZeroVelocities(true);

    testing::internal::CaptureStdout();
    const auto logContent = _writeSetupInfoAndReadLog();
    const auto stdOut     = testing::internal::GetCapturedStdout();

    EXPECT_NE(logContent.find(ZERO_VELOCITY_WARNING), std::string::npos);
    EXPECT_NE(stdOut.find(ZERO_VELOCITY_WARNING), std::string::npos);
}

TEST_F(TestZeroVelocityWarning, SilentWithInitOn)
{
    _settings->simulationBoxSettings.setInitializeVelocsByUser(
        InitVelocities::TRUE
    );
    setup::SimulationBoxSetup::setZeroVelocities(true);

    const auto logContent = _writeSetupInfoAndReadLog();

    EXPECT_EQ(logContent.find(ZERO_VELOCITY_WARNING), std::string::npos);
}

TEST_F(TestZeroVelocityWarning, SilentWithNonZeroVelocity)
{
    auto &simBox = _engine->get_simulationBox();

    const auto atom1 = std::make_shared<::molsys::Atom>();
    const auto atom2 = std::make_shared<::molsys::Atom>();
    atom2->setVelocity(linalg::Vec3D{0.0, 0.0, 1.0} * angstrom / s);
    simBox.getAtoms().push_back(atom1);
    simBox.getAtoms().push_back(atom2);

    setup::SimulationBoxSetup simBoxSetup(*_engine);
    simBoxSetup.checkZeroVelocities();

    const auto logContent = _writeSetupInfoAndReadLog();

    EXPECT_EQ(logContent.find(ZERO_VELOCITY_WARNING), std::string::npos);
}

TEST_F(TestZeroVelocityWarning, SilentWithZeroStartTemperature)
{
    _settings->kinetics.setStartTemperatureByUser(Temperature::zero());
    setup::SimulationBoxSetup::setZeroVelocities(true);

    const auto logContent = _writeSetupInfoAndReadLog();

    EXPECT_EQ(logContent.find(ZERO_VELOCITY_WARNING), std::string::npos);
}

TEST_F(TestZeroVelocityWarning, SilentWithoutTemperature)
{
    _settings->kinetics = KineticsSettings();
    setup::SimulationBoxSetup::setZeroVelocities(true);

    const auto logContent = _writeSetupInfoAndReadLog();

    EXPECT_EQ(logContent.find(ZERO_VELOCITY_WARNING), std::string::npos);
}

TEST_F(TestZeroVelocityWarning, SilentForNonMDJob)
{
    _settings->generalSettings.setJobtypeByUser(JobType::MM_OPT);
    setup::SimulationBoxSetup::setZeroVelocities(true);

    const auto logContent = _writeSetupInfoAndReadLog();

    EXPECT_EQ(logContent.find(ZERO_VELOCITY_WARNING), std::string::npos);
}

TEST_F(TestZeroVelocityWarning, SilentForRingPolymerMD)
{
    _settings->generalSettings.setJobtypeByUser(JobType::RING_POLYMER_QM_MD);
    setup::SimulationBoxSetup::setZeroVelocities(true);

    const auto logContent = _writeSetupInfoAndReadLog();

    EXPECT_EQ(logContent.find(ZERO_VELOCITY_WARNING), std::string::npos);
}
