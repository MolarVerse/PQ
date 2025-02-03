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

#include "testSimulationBox.hpp"

#include <algorithm>   // for copy
#include <cstddef>     // for size_t, std
#include <map>         // for map
#include <optional>    // for optional
#include <string>      // for string
#include <vector>      // for vector

#include "exceptions.hpp"   // for ManostatException, RstFileException
#include "gtest/gtest.h"    // for Message, TestPartResult, AssertionRe...
#include "potentialSettings.hpp"   // for PotentialSettings
#include "throwWithMessage.hpp"    // for throwWithMessage

/**
 * @brief tests numberOfAtoms function
 *
 */
TEST_F(TestSimulationBox, numberOfAtoms)
{
    EXPECT_EQ(_simulationBox->getNumberOfAtoms(), 5);
}

/**
 * @brief tests calculateDegreesOfFreedom function
 *
 */
TEST_F(TestSimulationBox, calculateDegreesOfFreedom)
{
    _simulationBox->calculateDegreesOfFreedom();
    EXPECT_EQ(_simulationBox->getDegreesOfFreedom(), 12);
}

/**
 * @brief tests calculateTotalForce function
 *
 */
TEST_F(TestSimulationBox, calculateTotalForce)
{
    auto totalForce = _simulationBox->calculateTotalForce();

    EXPECT_NEAR(totalForce, 1.7320508075688772, 1e-8);
}

/**
 * @brief tests calculateTotalForce function
 *
 */
TEST_F(TestSimulationBox, calculateTotalForceVector)
{
    auto totalForceVector = _simulationBox->calculateTotalForceVector();

    EXPECT_EQ(totalForceVector, linearAlgebra::Vec3D({1.0, 1.0, 1.0}));
}

/**
 * @brief tests calculateCenterOfMass function
 *
 */
TEST_F(TestSimulationBox, centerOfMassOfMolecules)
{
    _simulationBox->calculateCenterOfMassMolecules();

    auto molecules = _simulationBox->getMolecules();

    EXPECT_EQ(
        molecules[0].getCenterOfMass(),
        linearAlgebra::Vec3D(1 / 3.0, 0.5, 0.0)
    );
    EXPECT_EQ(
        molecules[1].getCenterOfMass(),
        linearAlgebra::Vec3D(2 / 3.0, 0.0, 0.0)
    );
}

/**
 * @brief tests findMoleculeType function
 *
 */
TEST_F(TestSimulationBox, findMolecule)
{
    auto molecule = _simulationBox->findMolecule(1);
    EXPECT_EQ(molecule.value().getMoltype(), 1);

    molecule = _simulationBox->findMolecule(3);
    EXPECT_EQ(molecule, std::nullopt);
}

/**
 * @brief tests findMoleculeType function
 *
 */
TEST_F(TestSimulationBox, findMoleculeType)
{
    const auto molecule = _simulationBox->findMoleculeType(1);
    EXPECT_EQ(molecule.getMoltype(), 1);

    EXPECT_THROW(
        [[maybe_unused]] auto &dummy = _simulationBox->findMoleculeType(3),
        customException::RstFileException
    );
}

/**
 * @brief tests findMoleculeByAtomIndex function
 *
 */
TEST_F(TestSimulationBox, findMoleculeByAtomIndex)
{
    const auto &[molecule1, atomIndex1] =
        _simulationBox->findMoleculeByAtomIndex(3);
    EXPECT_EQ(molecule1, &(_simulationBox->getMolecules()[0]));
    EXPECT_EQ(atomIndex1, 2);

    const auto &[molecule2, atomIndex2] =
        _simulationBox->findMoleculeByAtomIndex(4);
    EXPECT_EQ(molecule2, &(_simulationBox->getMolecules()[1]));
    EXPECT_EQ(atomIndex2, 0);

    EXPECT_THROW([[maybe_unused]] const auto dummy =
                     _simulationBox->findMoleculeByAtomIndex(6);
                 , customException::UserInputException);
}

/**
 * @brief tests findNecessaryMoleculeTypes function
 *
 */
TEST_F(TestSimulationBox, findNecessaryMoleculeTypes)
{
    auto simulationBox = simulationBox::SimulationBox();
    auto molecule1     = simulationBox::Molecule();
    auto molecule2     = simulationBox::Molecule();
    auto molecule3     = simulationBox::Molecule();

    molecule1.setMoltype(1);
    molecule2.setMoltype(2);
    molecule3.setMoltype(3);

    simulationBox.addMolecule(molecule1);
    simulationBox.addMolecule(molecule2);
    simulationBox.addMolecule(molecule3);
    simulationBox.addMolecule(molecule2);
    simulationBox.addMolecule(molecule1);

    const auto moleculeType1 = simulationBox::MoleculeType(1);
    const auto moleculeType2 = simulationBox::MoleculeType(2);
    const auto moleculeType3 = simulationBox::MoleculeType(3);

    simulationBox.addMoleculeType(moleculeType1);
    simulationBox.addMoleculeType(moleculeType2);
    simulationBox.addMoleculeType(moleculeType3);

    auto necessaryMoleculeTypes = simulationBox.findNecessaryMoleculeTypes();
    EXPECT_EQ(necessaryMoleculeTypes.size(), 3);
    EXPECT_EQ(necessaryMoleculeTypes[0].getMoltype(), 1);
    EXPECT_EQ(necessaryMoleculeTypes[1].getMoltype(), 2);
    EXPECT_EQ(necessaryMoleculeTypes[2].getMoltype(), 3);
}

/**
 * @brief tests checkCoulombRadiusCutoff function if the radius cut off is
 * larger than half of the minimal box
 */
TEST_F(TestSimulationBox, checkCoulombRadiusCutoff)
{
    settings::PotentialSettings::setCoulombRadiusCutOff(1.0);
    _simulationBox->setBoxDimensions({1.99, 10.0, 10.0});

    EXPECT_THROW_MSG(
        _simulationBox->checkCoulRadiusCutOff(
            customException::ExceptionType::USERINPUTEXCEPTION
        ),
        customException::UserInputException,
        "Coulomb radius cut off is larger than half of the minimal box "
        "dimension"
    );

    EXPECT_THROW_MSG(
        _simulationBox->checkCoulRadiusCutOff(
            customException::ExceptionType::MANOSTATEXCEPTION
        ),
        customException::ManostatException,
        "Coulomb radius cut off is larger than half of the minimal box "
        "dimension"
    );

    _simulationBox->setBoxDimensions({10.0, 1.99, 10.0});

    EXPECT_THROW_MSG(
        _simulationBox->checkCoulRadiusCutOff(
            customException::ExceptionType::USERINPUTEXCEPTION
        ),
        customException::UserInputException,
        "Coulomb radius cut off is larger than half of the minimal box "
        "dimension"
    );

    _simulationBox->setBoxDimensions({10.0, 10.0, 1.99});

    EXPECT_THROW_MSG(
        _simulationBox->checkCoulRadiusCutOff(
            customException::ExceptionType::USERINPUTEXCEPTION
        ),
        customException::UserInputException,
        "Coulomb radius cut off is larger than half of the minimal box "
        "dimension"
    );
}

/**
 * @brief tests setup external to internal global vdw types map
 *
 */
TEST_F(TestSimulationBox, setupExternalToInternalGlobalVdwTypesMap)
{
    simulationBox::SimulationBox simulationBox;
    simulationBox::MoleculeType  molecule1(1);
    simulationBox::MoleculeType  molecule2(2);

    molecule1.addExternalGlobalVDWType(1);
    molecule1.addExternalGlobalVDWType(3);
    molecule1.addExternalGlobalVDWType(5);

    molecule2.addExternalGlobalVDWType(3);
    molecule2.addExternalGlobalVDWType(5);

    simulationBox.addMoleculeType(molecule1);
    simulationBox.addMoleculeType(molecule2);

    simulationBox.setupExternalToInternalGlobalVdwTypesMap();

    EXPECT_EQ(simulationBox.getExternalGlobalVdwTypes().size(), 3);
    EXPECT_EQ(
        simulationBox.getExternalGlobalVdwTypes(),
        std::vector<size_t>({1, 3, 5})
    );

    EXPECT_EQ(simulationBox.getExternalToInternalGlobalVDWTypes().size(), 3);
    EXPECT_EQ(simulationBox.getExternalToInternalGlobalVDWTypes().at(1), 0);
    EXPECT_EQ(simulationBox.getExternalToInternalGlobalVDWTypes().at(3), 1);
    EXPECT_EQ(simulationBox.getExternalToInternalGlobalVDWTypes().at(5), 2);
}

/**
 * @brief tests moleculeTypeExists function
 *
 */
TEST_F(TestSimulationBox, moleculeTypeExists)
{
    _simulationBox->getMoleculeTypes()[0].setMoltype(1);
    _simulationBox->getMoleculeTypes()[1].setMoltype(2);

    EXPECT_TRUE(_simulationBox->moleculeTypeExists(1));
    EXPECT_FALSE(_simulationBox->moleculeTypeExists(3));
}

/**
 * @brief tests findMoleculeTypeByString function
 *
 * @details findMoleculeTypeByString returns an optional size_t.
 *
 */
TEST_F(TestSimulationBox, findMoleculeTypeByString)
{
    _simulationBox->getMoleculeTypes()[0].setName("mol1");
    _simulationBox->getMoleculeTypes()[1].setName("mol2");

    EXPECT_EQ(_simulationBox->findMoleculeTypeByString("mol1").value(), 1);
    EXPECT_EQ(_simulationBox->findMoleculeTypeByString("mol2").value(), 2);
    EXPECT_EQ(
        _simulationBox->findMoleculeTypeByString("mol3").has_value(),
        false
    );
}

/**
 * @brief tests setPartialChargesOfMoleculesFromMoleculeTypes function
 *
 */
TEST_F(TestSimulationBox, setPartialChargesOfMoleculesFromMoleculeTypes)
{
    simulationBox::SimulationBox simulationBox;
    simulationBox::MoleculeType  molecule1(1);
    simulationBox::MoleculeType  molecule2(2);

    molecule1.setPartialCharges({0.1, 0.2, 0.3});
    molecule2.setPartialCharges({0.4, 0.5});

    const auto atom1 = std::make_shared<simulationBox::Atom>();
    const auto atom2 = std::make_shared<simulationBox::Atom>();
    const auto atom3 = std::make_shared<simulationBox::Atom>();
    const auto atom4 = std::make_shared<simulationBox::Atom>();
    const auto atom5 = std::make_shared<simulationBox::Atom>();
    const auto atom6 = std::make_shared<simulationBox::Atom>();
    const auto atom7 = std::make_shared<simulationBox::Atom>();
    const auto atom8 = std::make_shared<simulationBox::Atom>();

    simulationBox::Molecule molecule3(1);
    simulationBox::Molecule molecule4(2);
    simulationBox::Molecule molecule5(1);

    molecule3.setNumberOfAtoms(3);
    molecule4.setNumberOfAtoms(2);
    molecule5.setNumberOfAtoms(3);

    molecule3.addAtom(atom1);
    molecule3.addAtom(atom2);
    molecule3.addAtom(atom3);
    molecule4.addAtom(atom4);
    molecule4.addAtom(atom5);
    molecule5.addAtom(atom6);
    molecule5.addAtom(atom7);
    molecule5.addAtom(atom8);

    simulationBox.addMoleculeType(molecule1);
    simulationBox.addMoleculeType(molecule2);

    simulationBox.addMolecule(molecule3);
    simulationBox.addMolecule(molecule4);
    simulationBox.addMolecule(molecule5);

    simulationBox.setPartialChargesOfMoleculesFromMoleculeTypes();

    EXPECT_EQ(
        simulationBox.getMolecule(0).getPartialCharges(),
        molecule1.getPartialCharges()
    );
    EXPECT_EQ(
        simulationBox.getMolecule(1).getPartialCharges(),
        molecule2.getPartialCharges()
    );
    EXPECT_EQ(
        simulationBox.getMolecule(2).getPartialCharges(),
        molecule1.getPartialCharges()
    );
}

/**
 * @brief tests setPartialChargesOfMoleculesFromMoleculeTypes function
 *
 */
TEST_F(
    TestSimulationBox,
    setPartialChargesOfMoleculesFromMoleculeTypes_MoleculeTypeNotFound
)
{
    simulationBox::SimulationBox  simulationBox;
    const simulationBox::Molecule molecule1(1);

    simulationBox.addMolecule(molecule1);

    EXPECT_THROW_MSG(
        simulationBox.setPartialChargesOfMoleculesFromMoleculeTypes(),
        customException::UserInputException,
        "Molecule type 1 not found in molecule types"
    );
}