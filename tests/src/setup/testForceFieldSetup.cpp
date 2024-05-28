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

#include "angleForceField.hpp"      // for AngleForceField
#include "angleType.hpp"            // for AngleType
#include "bondForceField.hpp"       // for BondForceField
#include "bondType.hpp"             // for BondType
#include "dihedralForceField.hpp"   // for DihedralForceField
#include "dihedralType.hpp"         // for DihedralType
#include "engine.hpp"               // for Engine
#include "forceFieldClass.hpp"      // for ForceField
#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "forceFieldSetup.hpp"      // for ForceFieldSetup, setupForceField
#include "molecule.hpp"             // for Molecule
#include "simulationBox.hpp"        // for SimulationBox
#include "testSetup.hpp"            // for TestSetup

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cstddef>         // for size_t
#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)
#include <vector>          // for vector, allocator

/**
 * @brief test setupBonds function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupBonds)
{
    auto molecule1 = simulationBox::Molecule();
    auto molecule2 = simulationBox::Molecule();

    _engine->getSimulationBox().addMolecule(molecule1);
    _engine->getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine->getSimulationBox().getMolecule(1);

    auto bond1 = forceField::BondForceField(molecule1Ptr, molecule2Ptr, 0, 1, 0);
    auto bond2 = forceField::BondForceField(molecule1Ptr, molecule1Ptr, 0, 1, 1);
    auto bond3 = forceField::BondForceField(molecule1Ptr, molecule2Ptr, 0, 1, 0);

    _engine->getForceFieldPtr()->addBond(bond1);
    _engine->getForceFieldPtr()->addBond(bond2);
    _engine->getForceFieldPtr()->addBond(bond3);

    auto bondType1 = forceField::BondType(0, 1.0, 1.0);
    auto bondType2 = forceField::BondType(1, 2.0, 2.0);

    _engine->getForceFieldPtr()->addBondType(bondType1);
    _engine->getForceFieldPtr()->addBondType(bondType2);

    auto setup = setup::ForceFieldSetup(*_engine);
    setup.setupBonds();

    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[0].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[0].getEquilibriumBondLength(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[0].getForceConstant(), 1.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[1].getType(), 1);
    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[1].getEquilibriumBondLength(), 2.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[1].getForceConstant(), 2.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[2].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[2].getEquilibriumBondLength(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[2].getForceConstant(), 1.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getBondTypes().size(), 0);
}

/**
 * @brief test setupAngles function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupAngles)
{
    auto molecule1 = simulationBox::Molecule();
    auto molecule2 = simulationBox::Molecule();

    _engine->getSimulationBox().addMolecule(molecule1);
    _engine->getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine->getSimulationBox().getMolecule(1);

    auto angle1 = forceField::AngleForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2}, 0);
    auto angle2 = forceField::AngleForceField({molecule1Ptr, molecule1Ptr, molecule2Ptr}, {0, 1, 2}, 1);
    auto angle3 = forceField::AngleForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2}, 0);

    _engine->getForceFieldPtr()->addAngle(angle1);
    _engine->getForceFieldPtr()->addAngle(angle2);
    _engine->getForceFieldPtr()->addAngle(angle3);

    auto angleType1 = forceField::AngleType(0, 1.0, 1.0);
    auto angleType2 = forceField::AngleType(1, 2.0, 2.0);

    _engine->getForceFieldPtr()->addAngleType(angleType1);
    _engine->getForceFieldPtr()->addAngleType(angleType2);

    auto setup = setup::ForceFieldSetup(*_engine);
    setup.setupAngles();

    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[0].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[0].getEquilibriumAngle(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[0].getForceConstant(), 1.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[1].getType(), 1);
    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[1].getEquilibriumAngle(), 2.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[1].getForceConstant(), 2.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[2].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[2].getEquilibriumAngle(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[2].getForceConstant(), 1.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getAngleTypes().size(), 0);
}

/**
 * @brief test setupDihedrals function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupDihedrals)
{
    auto molecule1 = simulationBox::Molecule();
    auto molecule2 = simulationBox::Molecule();

    _engine->getSimulationBox().addMolecule(molecule1);
    _engine->getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine->getSimulationBox().getMolecule(1);

    auto dihedral1 = forceField::DihedralForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 0);
    auto dihedral2 = forceField::DihedralForceField({molecule1Ptr, molecule1Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 1);
    auto dihedral3 = forceField::DihedralForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 0);

    _engine->getForceFieldPtr()->addDihedral(dihedral1);
    _engine->getForceFieldPtr()->addDihedral(dihedral2);
    _engine->getForceFieldPtr()->addDihedral(dihedral3);

    auto dihedralType1 = forceField::DihedralType(0, 1.0, 1.0, 1.0);
    auto dihedralType2 = forceField::DihedralType(1, 2.0, 2.0, 2.0);

    _engine->getForceFieldPtr()->addDihedralType(dihedralType1);
    _engine->getForceFieldPtr()->addDihedralType(dihedralType2);

    auto setup = setup::ForceFieldSetup(*_engine);
    setup.setupDihedrals();

    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[0].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[0].getForceConstant(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[0].getPhaseShift(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[0].getPeriodicity(), 1.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[1].getType(), 1);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[1].getForceConstant(), 2.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[1].getPhaseShift(), 2.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[1].getPeriodicity(), 2.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[2].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[2].getForceConstant(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[2].getPhaseShift(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[2].getPeriodicity(), 1.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedralTypes().size(), 0);
}

/**
 * @brief test setupImproperDihedrals function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupImproperDihedrals)
{
    auto molecule1 = simulationBox::Molecule();
    auto molecule2 = simulationBox::Molecule();

    _engine->getSimulationBox().addMolecule(molecule1);
    _engine->getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine->getSimulationBox().getMolecule(1);

    auto dihedral1 = forceField::DihedralForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 0);
    auto dihedral2 = forceField::DihedralForceField({molecule1Ptr, molecule1Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 1);
    auto dihedral3 = forceField::DihedralForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 0);

    _engine->getForceFieldPtr()->addImproperDihedral(dihedral1);
    _engine->getForceFieldPtr()->addImproperDihedral(dihedral2);
    _engine->getForceFieldPtr()->addImproperDihedral(dihedral3);

    auto dihedralType1 = forceField::DihedralType(0, 1.0, 1.0, 1.0);
    auto dihedralType2 = forceField::DihedralType(1, 2.0, 2.0, 2.0);

    _engine->getForceFieldPtr()->addImproperDihedralType(dihedralType1);
    _engine->getForceFieldPtr()->addImproperDihedralType(dihedralType2);

    auto setup = setup::ForceFieldSetup(*_engine);
    setup.setupImproperDihedrals();

    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getForceConstant(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getPhaseShift(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getPeriodicity(), 1.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[1].getType(), 1);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[1].getForceConstant(), 2.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[1].getPhaseShift(), 2.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[1].getPeriodicity(), 2.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[2].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[2].getForceConstant(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[2].getPhaseShift(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[2].getPeriodicity(), 1.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedralTypes().size(), 0);
}

/**
 * @brief test setupForceField function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupForceField)
{
    settings::ForceFieldSettings::activate();

    auto molecule1 = simulationBox::Molecule();
    _engine->getSimulationBox().addMolecule(molecule1);
    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);

    auto bond     = forceField::BondForceField(molecule1Ptr, molecule1Ptr, 0, 1, 0);
    auto angle    = forceField::AngleForceField({molecule1Ptr, molecule1Ptr, molecule1Ptr}, {0, 1, 2}, 0);
    auto dihedral = forceField::DihedralForceField({molecule1Ptr, molecule1Ptr, molecule1Ptr, molecule1Ptr}, {0, 1, 2, 3}, 0);
    auto improperDihedral =
        forceField::DihedralForceField({molecule1Ptr, molecule1Ptr, molecule1Ptr, molecule1Ptr}, {0, 1, 2, 3}, 0);

    _engine->getForceFieldPtr()->addBond(bond);
    _engine->getForceFieldPtr()->addAngle(angle);
    _engine->getForceFieldPtr()->addDihedral(dihedral);
    _engine->getForceFieldPtr()->addImproperDihedral(improperDihedral);

    auto bondType             = forceField::BondType(0, 1.0, 2.0);
    auto angleType            = forceField::AngleType(0, 2.0, 3.0);
    auto dihedralType         = forceField::DihedralType(0, 3.0, 4.0, 5.0);
    auto improperDihedralType = forceField::DihedralType(0, 4.0, 5.0, 6.0);

    _engine->getForceFieldPtr()->addBondType(bondType);
    _engine->getForceFieldPtr()->addAngleType(angleType);
    _engine->getForceFieldPtr()->addDihedralType(dihedralType);
    _engine->getForceFieldPtr()->addImproperDihedralType(improperDihedralType);

    setup::setupForceField(*_engine);

    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[0].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[0].getEquilibriumBondLength(), 1.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getBonds()[0].getForceConstant(), 2.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[0].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[0].getEquilibriumAngle(), 2.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getAngles()[0].getForceConstant(), 3.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[0].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[0].getForceConstant(), 3.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[0].getPeriodicity(), 4.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getDihedrals()[0].getPhaseShift(), 5.0);

    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getType(), 0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getForceConstant(), 4.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getPeriodicity(), 5.0);
    EXPECT_EQ(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getPhaseShift(), 6.0);
}

/**
 * @brief setupForceField should do nothing if force field is not activated
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupForceField_doNothing)
{
    settings::ForceFieldSettings::activate();

    auto molecule1 = simulationBox::Molecule();
    _engine->getSimulationBox().addMolecule(molecule1);
    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);

    auto bond     = forceField::BondForceField(molecule1Ptr, molecule1Ptr, 0, 1, 0);
    auto angle    = forceField::AngleForceField({molecule1Ptr, molecule1Ptr, molecule1Ptr}, {0, 1, 2}, 0);
    auto dihedral = forceField::DihedralForceField({molecule1Ptr, molecule1Ptr, molecule1Ptr, molecule1Ptr}, {0, 1, 2, 3}, 0);
    auto improperDihedral =
        forceField::DihedralForceField({molecule1Ptr, molecule1Ptr, molecule1Ptr, molecule1Ptr}, {0, 1, 2, 3}, 0);

    _engine->getForceFieldPtr()->addBond(bond);
    _engine->getForceFieldPtr()->addAngle(angle);
    _engine->getForceFieldPtr()->addDihedral(dihedral);
    _engine->getForceFieldPtr()->addImproperDihedral(improperDihedral);

    auto bondType             = forceField::BondType(0, 1.0, 2.0);
    auto angleType            = forceField::AngleType(0, 2.0, 3.0);
    auto dihedralType         = forceField::DihedralType(0, 3.0, 4.0, 5.0);
    auto improperDihedralType = forceField::DihedralType(0, 4.0, 5.0, 6.0);

    _engine->getForceFieldPtr()->addBondType(bondType);
    _engine->getForceFieldPtr()->addAngleType(angleType);
    _engine->getForceFieldPtr()->addDihedralType(dihedralType);
    _engine->getForceFieldPtr()->addImproperDihedralType(improperDihedralType);

    settings::ForceFieldSettings::deactivate();
    setup::setupForceField(*_engine);

    EXPECT_NE(_engine->getForceFieldPtr()->getBonds()[0].getEquilibriumBondLength(), 1.0);
    EXPECT_NE(_engine->getForceFieldPtr()->getBonds()[0].getForceConstant(), 2.0);

    EXPECT_NE(_engine->getForceFieldPtr()->getAngles()[0].getEquilibriumAngle(), 2.0);
    EXPECT_NE(_engine->getForceFieldPtr()->getAngles()[0].getForceConstant(), 3.0);

    EXPECT_NE(_engine->getForceFieldPtr()->getDihedrals()[0].getForceConstant(), 3.0);
    EXPECT_NE(_engine->getForceFieldPtr()->getDihedrals()[0].getPeriodicity(), 4.0);
    EXPECT_NE(_engine->getForceFieldPtr()->getDihedrals()[0].getPhaseShift(), 5.0);

    EXPECT_NE(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getForceConstant(), 4.0);
    EXPECT_NE(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getPeriodicity(), 5.0);
    EXPECT_NE(_engine->getForceFieldPtr()->getImproperDihedrals()[0].getPhaseShift(), 6.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
