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

#include <cstddef>    // for size_t, std
#include <map>        // for map
#include <optional>   // for optional
#include <string>     // for string
#include <vector>     // for vector

#include "exceptions.hpp"   // for ManostatException, RstFileException
#include "gtest/gtest.h"    // for Message, TestPartResult, AssertionRe...
#include "potentialSettings.hpp"   // for PotentialSettings
#include "throwWithMessage.hpp"    // for throwWithMessage
#include "vectorNear.hpp"          // for EXPECT_VECTOR_NEAR

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
        exc::RstFileException
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
                 , exc::UserInputException);

    EXPECT_THROW([[maybe_unused]] const auto dummy =
                     _simulationBox->findMoleculeByAtomIndex(0);
                 , exc::UserInputException);
}

/**
 * @brief tests findNecessaryMoleculeTypes function
 *
 */
TEST_F(TestSimulationBox, findNecessaryMoleculeTypes)
{
    auto simulationBox = molsys::SimulationBox();
    auto molecule1     = molsys::Molecule();
    auto molecule2     = molsys::Molecule();
    auto molecule3     = molsys::Molecule();

    molecule1.setMoltype(1);
    molecule2.setMoltype(2);
    molecule3.setMoltype(3);

    simulationBox.addMolecule(molecule1);
    simulationBox.addMolecule(molecule2);
    simulationBox.addMolecule(molecule3);
    simulationBox.addMolecule(molecule2);
    simulationBox.addMolecule(molecule1);

    const auto moleculeType1 = molsys::MoleculeType(1);
    const auto moleculeType2 = molsys::MoleculeType(2);
    const auto moleculeType3 = molsys::MoleculeType(3);

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
        _simulationBox->checkCoulRadiusCutOff(ExceptionType::UserInputError),
        exc::UserInputException,
        "Coulomb radius cut off is larger than half of the minimal box "
        "dimension"
    );

    EXPECT_THROW_MSG(
        _simulationBox->checkCoulRadiusCutOff(ExceptionType::ManostatError),
        exc::ManostatException,
        "Coulomb radius cut off is larger than half of the minimal box "
        "dimension"
    );

    _simulationBox->setBoxDimensions({10.0, 1.99, 10.0});

    EXPECT_THROW_MSG(
        _simulationBox->checkCoulRadiusCutOff(ExceptionType::UserInputError),
        exc::UserInputException,
        "Coulomb radius cut off is larger than half of the minimal box "
        "dimension"
    );

    _simulationBox->setBoxDimensions({10.0, 10.0, 1.99});

    EXPECT_THROW_MSG(
        _simulationBox->checkCoulRadiusCutOff(ExceptionType::UserInputError),
        exc::UserInputException,
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
    molsys::SimulationBox simulationBox;
    molsys::MoleculeType  molecule1(1);
    molsys::MoleculeType  molecule2(2);

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
    molsys::SimulationBox simulationBox;
    molsys::MoleculeType  molecule1(1);
    molsys::MoleculeType  molecule2(2);

    molecule1.setPartialCharges({0.1, 0.2, 0.3});
    molecule2.setPartialCharges({0.4, 0.5});

    const auto atom1 = std::make_shared<molsys::Atom>();
    const auto atom2 = std::make_shared<molsys::Atom>();
    const auto atom3 = std::make_shared<molsys::Atom>();
    const auto atom4 = std::make_shared<molsys::Atom>();
    const auto atom5 = std::make_shared<molsys::Atom>();
    const auto atom6 = std::make_shared<molsys::Atom>();
    const auto atom7 = std::make_shared<molsys::Atom>();
    const auto atom8 = std::make_shared<molsys::Atom>();

    molsys::Molecule molecule3(1);
    molsys::Molecule molecule4(2);
    molsys::Molecule molecule5(1);

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
    setPartialChargesOfMoleculesFromMoleculeTypesMoleculeTypeNotFound
)
{
    molsys::SimulationBox  simulationBox;
    const molsys::Molecule molecule1(1);

    simulationBox.addMolecule(molecule1);

    EXPECT_THROW_MSG(
        simulationBox.setPartialChargesOfMoleculesFromMoleculeTypes(),
        exc::UserInputException,
        "Molecule type 1 not found in molecule types"
    );
}

/**
 * @brief tests molsys::removeNetForce()
 *
 */
TEST_F(TestSimulationBox, removeNetForce)
{
    using namespace molsys;
    using namespace linearAlgebra;

    SimulationBox simBox;
    auto          atom1 = Atom();
    auto          atom2 = Atom();
    auto          atom3 = Atom();

    atom1.setForce({3.0, 1.0, 0.0});
    atom2.setForce({2.0, 4.0, -2.0});
    atom3.setForce({1.0, 4.0, 2.0});

    simBox.addAtom(std::make_shared<Atom>(atom1));
    simBox.addAtom(std::make_shared<Atom>(atom2));
    simBox.addAtom(std::make_shared<Atom>(atom3));

    EXPECT_VECTOR_NEAR(
        simBox.calculateTotalForceVector(),
        Vec3D({6.0, 9.0, 0.0}),
        1e-10
    );

    simBox.removeNetForce();

    EXPECT_VECTOR_NEAR(
        simBox.calculateTotalForceVector(),
        Vec3D({0.0, 0.0, 0.0}),
        1e-10
    );

    EXPECT_VECTOR_NEAR(
        simBox.getAtom(0).getForce(),
        Vec3D({1.0, -2.0, 0.0}),
        1e-10
    );
    EXPECT_VECTOR_NEAR(
        simBox.getAtom(1).getForce(),
        Vec3D({0.0, 1.0, -2.0}),
        1e-10
    );
    EXPECT_VECTOR_NEAR(
        simBox.getAtom(2).getForce(),
        Vec3D({-1.0, 1.0, 2.0}),
        1e-10
    );
}

/**
 * @brief tests molsys::updateOldPositions()
 *
 */
TEST_F(TestSimulationBox, updateOldPositions)
{
    using namespace molsys;
    using namespace linearAlgebra;

    _simulationBox->getAtoms()[0]->setPositionOld({9.0, 9.0, 9.0});
    _simulationBox->getAtoms()[1]->setPositionOld({9.0, 9.0, 9.0});

    _simulationBox->updateOldPositions();

    for (const auto &atom : _simulationBox->getAtoms())
    {
        EXPECT_VECTOR_NEAR(atom->getPositionOld(), atom->getPosition(), 1e-10);
    }
}

TEST_F(TestSimulationBox, copyOwnsIndependentAtoms)
{
    molsys::SimulationBox copied;
    copied.copy(*_simulationBox);

    ASSERT_EQ(copied.getNumberOfAtoms(), _simulationBox->getNumberOfAtoms());
    ASSERT_EQ(
        copied.getNumberOfMolecules(),
        _simulationBox->getNumberOfMolecules()
    );
    EXPECT_NE(&copied.getAtom(0), &_simulationBox->getAtom(0));
    EXPECT_EQ(&copied.getMolecule(0).getAtom(0), &copied.getAtom(0));

    const auto cloned = _simulationBox->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->getNumberOfAtoms(), _simulationBox->getNumberOfAtoms());
}

TEST_F(TestSimulationBox, validatesHybridIndexLists)
{
    _simulationBox->addInnerRegionCenterAtoms({0, 4});
    EXPECT_EQ(
        _simulationBox->getInnerRegionCenterAtomIndices(),
        std::vector<int>({0, 4})
    );
    EXPECT_THROW(
        _simulationBox->addInnerRegionCenterAtoms({-1}),
        exc::UserInputException
    );
    EXPECT_THROW(
        _simulationBox->addInnerRegionCenterAtoms({5}),
        exc::UserInputException
    );

    _simulationBox->setupForcedOuterMolecules({0});
    EXPECT_TRUE(_simulationBox->getMolecule(0).isForcedOuter());
    EXPECT_THROW(
        _simulationBox->setupForcedCoreMolecules({0}),
        exc::UserInputException
    );

    _simulationBox->setupForcedCoreMolecules({1});
    EXPECT_TRUE(_simulationBox->getMolecule(1).isForcedCore());
    EXPECT_THROW(
        _simulationBox->setupForcedOuterMolecules({1}),
        exc::UserInputException
    );
    EXPECT_THROW(
        _simulationBox->setupForcedCoreMolecules({2}),
        exc::UserInputException
    );
    EXPECT_THROW(
        _simulationBox->setupForcedOuterMolecules({-1}),
        exc::UserInputException
    );
}

TEST_F(TestSimulationBox, validatesForcedLayerList)
{
    molsys::SimulationBox simBox;
    simBox.addMolecule(molsys::Molecule{});
    simBox.addMolecule(molsys::Molecule{});
    simBox.addMolecule(molsys::Molecule{});

    simBox.setupForcedCoreMolecules({0});
    simBox.setupForcedLayerMolecules({1});
    simBox.setupForcedOuterMolecules({2});

    EXPECT_TRUE(simBox.getMolecule(0).isForcedCore());
    EXPECT_TRUE(simBox.getMolecule(1).isForcedLayer());
    EXPECT_TRUE(simBox.getMolecule(2).isForcedOuter());

    EXPECT_THROW_MSG(
        simBox.setupForcedLayerMolecules({-1}),
        exc::UserInputException,
        "Forced Layer region molecule index -1 out of range"
    );
    EXPECT_THROW_MSG(
        simBox.setupForcedLayerMolecules({3}),
        exc::UserInputException,
        "Forced Layer region molecule index 3 out of range"
    );
    EXPECT_THROW_MSG(
        simBox.setupForcedLayerMolecules({0}),
        exc::UserInputException,
        "Ambiguous molecule index 0 - molecule cannot be in "
        "forced_layer_list AND forced_core_list/forced_outer_list at the same "
        "time"
    );
    EXPECT_THROW_MSG(
        simBox.setupForcedLayerMolecules({2}),
        exc::UserInputException,
        "Ambiguous molecule index 2 - molecule cannot be in "
        "forced_layer_list AND forced_core_list/forced_outer_list at the same "
        "time"
    );
    EXPECT_THROW_MSG(
        simBox.setupForcedCoreMolecules({1}),
        exc::UserInputException,
        "Ambiguous molecule index 1 - molecule cannot be in "
        "forced_core_list AND forced_layer_list/forced_outer_list at the same "
        "time"
    );
    EXPECT_THROW_MSG(
        simBox.setupForcedOuterMolecules({1}),
        exc::UserInputException,
        "Ambiguous molecule index 1 - molecule cannot be in "
        "forced_outer_list AND forced_core_list/forced_layer_list at the same "
        "time"
    );
}

TEST_F(TestSimulationBox, assignsInternalVdwTypesToAtoms)
{
    molsys::SimulationBox simBox;
    molsys::MoleculeType  type(1);
    type.addExternalGlobalVDWType(4);
    type.addExternalGlobalVDWType(9);
    simBox.addMoleculeType(type);

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();
    atom1->setExternalGlobalVDWType(4);
    atom2->setExternalGlobalVDWType(9);

    molsys::Molecule molecule(1);
    molecule.setNumberOfAtoms(2);
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    simBox.addMolecule(molecule);

    simBox.setupExternalToInternalGlobalVdwTypesMap();

    EXPECT_EQ(simBox.getMolecule(0).getAtom(0).getInternalGlobalVDWType(), 0);
    EXPECT_EQ(simBox.getMolecule(0).getAtom(1).getInternalGlobalVDWType(), 1);
}

TEST_F(TestSimulationBox, forceMetricsAndAtomStateUpdates)
{
    using linearAlgebra::Vec3D;

    size_t index = 1;
    for (auto &atom : _simulationBox->getAtoms())
    {
        const auto value = static_cast<double>(index++);
        atom->setVelocity({value, 0.0, 0.0});
        atom->setForce({value, -value, 0.0});
        atom->setForceInner({value, 0.0, 0.0});
        atom->setForceOuter({0.0, value, 0.0});
        atom->setQMCharge(value);
    }

    _simulationBox->updateOldVelocities();
    _simulationBox->updateOldForces();

    EXPECT_GT(_simulationBox->calculateRMSForce(), 0.0);
    EXPECT_GT(_simulationBox->calculateMaxForce(), 0.0);
    EXPECT_GT(_simulationBox->calculateRMSForceOld(), 0.0);
    EXPECT_GT(_simulationBox->calculateMaxForceOld(), 0.0);

    _simulationBox->resetForcesInner();
    _simulationBox->resetForcesOuter();
    _simulationBox->resetQMCharges();
    for (const auto &atom : _simulationBox->getAtoms())
    {
        EXPECT_EQ(atom->getForceInner(), Vec3D{});
        EXPECT_EQ(atom->getForceOuter(), Vec3D{});
        EXPECT_FALSE(atom->getQMCharge().has_value());
    }

    for (auto &atom : _simulationBox->getAtoms())
    {
        atom->setForceInner({1.0, 0.0, 0.0});
        atom->setForceOuter({0.0, 1.0, 0.0});
    }
    _simulationBox->resetAllForces();
    for (const auto &atom : _simulationBox->getAtoms())
    {
        EXPECT_EQ(atom->getForce(), Vec3D{});
        EXPECT_EQ(atom->getForceInner(), Vec3D{});
        EXPECT_EQ(atom->getForceOuter(), Vec3D{});
    }

    const auto firstPosition = _simulationBox->getAtom(0).getPosition();
    _simulationBox->initPositions(0.0);
    EXPECT_EQ(_simulationBox->getAtom(0).getPosition(), firstPosition);
}

TEST_F(TestSimulationBox, activeChargeAndEmptyForceRemoval)
{
    _simulationBox->getMolecule(0).setCharge(2);
    _simulationBox->getMolecule(1).setCharge(-1);
    _simulationBox->getMolecule(1).deactivateMolecule();
    EXPECT_EQ(_simulationBox->calcActiveMolCharge(), 2);

    molsys::SimulationBox empty;
    EXPECT_NO_THROW(empty.removeNetForce());
}
