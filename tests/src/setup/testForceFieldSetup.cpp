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

#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)

#include <vector>   // for vector, allocator

#include "angleForceField.hpp"      // for AngleForceField
#include "angleType.hpp"            // for AngleType
#include "bondForceField.hpp"       // for BondForceField
#include "bondType.hpp"             // for BondType
#include "dihedralForceField.hpp"   // for DihedralForceField
#include "dihedralType.hpp"         // for DihedralType
#include "engine.hpp"               // for Engine
#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "forceFieldSetup.hpp"      // for ForceFieldSetup, setupForceField
#include "gtest/gtest.h"            // for Message, TestPartResult
#include "molecule.hpp"             // for Molecule
#include "strongTypes.hpp"
#include "testSetup.hpp"   // for TestSetup

/**
 * @brief test setupBonds function
 *
 */
TEST_F(TestSetup, forceFieldSetupSetupBonds)
{
    auto molecule1 = molsys::Molecule();
    auto molecule2 = molsys::Molecule();

    _engine->getSimulationBox().addMolecule(molecule1);
    _engine->getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine->getSimulationBox().getMolecule(1);

    auto bond1 = forceField::BondForceField(
        molecule1Ptr,
        molecule2Ptr,
        AtomIndex{0},
        AtomIndex{1},
        BondId{0}
    );
    auto bond2 = forceField::BondForceField(
        molecule1Ptr,
        molecule1Ptr,
        AtomIndex{0},
        AtomIndex{1},
        BondId{1}
    );
    auto bond3 = forceField::BondForceField(
        molecule1Ptr,
        molecule2Ptr,
        AtomIndex{0},
        AtomIndex{1},
        BondId{0}
    );

    _engine->getForceField()->addBond(bond1);
    _engine->getForceField()->addBond(bond2);
    _engine->getForceField()->addBond(bond3);

    auto bondType1 = forceField::BondType(
        BondId{0},
        BondParams{.equilibrium = 1.0, .forceConstant = 1.0}
    );
    auto bondType2 = forceField::BondType(
        BondId{1},
        BondParams{.equilibrium = 2.0, .forceConstant = 2.0}
    );

    _engine->getForceField()->addBondType(bondType1);
    _engine->getForceField()->addBondType(bondType2);

    auto setup = setup::ForceFieldSetup(*_engine);
    setup.setupBonds();

    const auto &bonds = _engine->getForceField()->getBonds();

    EXPECT_EQ(bonds[0].getType(), BondId{0});
    EXPECT_EQ(bonds[0].getParams().equilibrium, 1.0);
    EXPECT_EQ(bonds[0].getParams().forceConstant, 1.0);

    EXPECT_EQ(bonds[1].getType(), BondId{1});
    EXPECT_EQ(bonds[1].getParams().equilibrium, 2.0);
    EXPECT_EQ(bonds[1].getParams().forceConstant, 2.0);

    EXPECT_EQ(bonds[2].getType(), BondId{0});
    EXPECT_EQ(bonds[2].getParams().equilibrium, 1.0);
    EXPECT_EQ(bonds[2].getParams().forceConstant, 1.0);

    EXPECT_EQ(_engine->getForceField()->getBondTypes().size(), 0);
}

/**
 * @brief test setupAngles function
 *
 */
TEST_F(TestSetup, forceFieldSetupSetupAngles)
{
    auto molecule1 = molsys::Molecule();
    auto molecule2 = molsys::Molecule();

    _engine->getSimulationBox().addMolecule(molecule1);
    _engine->getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine->getSimulationBox().getMolecule(1);

    auto angle1 = forceField::AngleForceField(
        {molecule1Ptr, molecule2Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{0}
    );
    auto angle2 = forceField::AngleForceField(
        {molecule1Ptr, molecule1Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{1}
    );
    auto angle3 = forceField::AngleForceField(
        {molecule1Ptr, molecule2Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{0}
    );

    _engine->getForceField()->addAngle(angle1);
    _engine->getForceField()->addAngle(angle2);
    _engine->getForceField()->addAngle(angle3);

    auto angleType1 = forceField::AngleType(
        AngleId{0},
        AngleParams{.equilibrium = 1.0, .forceConstant = 1.0}
    );
    auto angleType2 = forceField::AngleType(
        AngleId{1},
        AngleParams{.equilibrium = 2.0, .forceConstant = 2.0}
    );

    _engine->getForceField()->addAngleType(angleType1);
    _engine->getForceField()->addAngleType(angleType2);

    auto setup = setup::ForceFieldSetup(*_engine);
    setup.setupAngles();

    const auto &angles = _engine->getForceField()->getAngles();

    EXPECT_EQ(angles[0].getType(), AngleId{0});
    EXPECT_EQ(angles[0].getParams().equilibrium, 1.0);
    EXPECT_EQ(angles[0].getParams().forceConstant, 1.0);

    EXPECT_EQ(angles[1].getType(), AngleId{1});
    EXPECT_EQ(angles[1].getParams().equilibrium, 2.0);
    EXPECT_EQ(angles[1].getParams().forceConstant, 2.0);

    EXPECT_EQ(angles[2].getType(), AngleId{0});
    EXPECT_EQ(angles[2].getParams().equilibrium, 1.0);
    EXPECT_EQ(angles[2].getParams().forceConstant, 1.0);

    EXPECT_EQ(_engine->getForceField()->getAngleTypes().size(), 0);
}

/**
 * @brief test setupDihedrals function
 *
 */
TEST_F(TestSetup, forceFieldSetupSetupDihedrals)
{
    auto molecule1 = molsys::Molecule();
    auto molecule2 = molsys::Molecule();

    _engine->getSimulationBox().addMolecule(molecule1);
    _engine->getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine->getSimulationBox().getMolecule(1);

    auto dihedral1 = forceField::DihedralForceField(
        {molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );
    auto dihedral2 = forceField::DihedralForceField(
        {molecule1Ptr, molecule1Ptr, molecule2Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{1}
    );
    auto dihedral3 = forceField::DihedralForceField(
        {molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );

    _engine->getForceField()->addDihedral(dihedral1);
    _engine->getForceField()->addDihedral(dihedral2);
    _engine->getForceField()->addDihedral(dihedral3);

    auto dihedralType1 = forceField::DihedralType(
        DihedralId{0},
        DihedralParams{
            .forceConstant = 1.0,
            .frequency     = 1.0,
            .phaseShift    = 1.0
        }
    );
    auto dihedralType2 = forceField::DihedralType(
        DihedralId{1},
        DihedralParams{
            .forceConstant = 2.0,
            .frequency     = 2.0,
            .phaseShift    = 2.0
        }
    );

    _engine->getForceField()->addDihedralType(dihedralType1);
    _engine->getForceField()->addDihedralType(dihedralType2);

    auto setup = setup::ForceFieldSetup(*_engine);
    setup.setupDihedrals();

    const auto &dihedrals = _engine->getForceField()->getDihedrals();

    EXPECT_EQ(dihedrals[0].getType(), DihedralId{0});
    EXPECT_EQ(dihedrals[0].getParams().forceConstant, 1.0);
    EXPECT_EQ(dihedrals[0].getParams().phaseShift, 1.0);
    EXPECT_EQ(dihedrals[0].getParams().frequency, 1.0);

    EXPECT_EQ(dihedrals[1].getType(), DihedralId{1});
    EXPECT_EQ(dihedrals[1].getParams().forceConstant, 2.0);
    EXPECT_EQ(dihedrals[1].getParams().phaseShift, 2.0);
    EXPECT_EQ(dihedrals[1].getParams().frequency, 2.0);

    EXPECT_EQ(dihedrals[2].getType(), DihedralId{0});
    EXPECT_EQ(dihedrals[2].getParams().forceConstant, 1.0);
    EXPECT_EQ(dihedrals[2].getParams().phaseShift, 1.0);
    EXPECT_EQ(dihedrals[2].getParams().frequency, 1.0);

    EXPECT_EQ(_engine->getForceField()->getDihedralTypes().size(), 0);
}

/**
 * @brief test setupImproperDihedrals function
 *
 */
TEST_F(TestSetup, forceFieldSetupSetupImproperDihedrals)
{
    auto molecule1 = molsys::Molecule();
    auto molecule2 = molsys::Molecule();

    _engine->getSimulationBox().addMolecule(molecule1);
    _engine->getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine->getSimulationBox().getMolecule(1);

    auto dihedral1 = forceField::DihedralForceField(
        {molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );
    auto dihedral2 = forceField::DihedralForceField(
        {molecule1Ptr, molecule1Ptr, molecule2Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{1}
    );
    auto dihedral3 = forceField::DihedralForceField(
        {molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );

    _engine->getForceField()->addImproperDihedral(dihedral1);
    _engine->getForceField()->addImproperDihedral(dihedral2);
    _engine->getForceField()->addImproperDihedral(dihedral3);

    auto dihedralType1 = forceField::DihedralType(
        DihedralId{0},
        DihedralParams{
            .forceConstant = 1.0,
            .frequency     = 1.0,
            .phaseShift    = 1.0
        }
    );
    auto dihedralType2 = forceField::DihedralType(
        DihedralId{1},
        DihedralParams{
            .forceConstant = 2.0,
            .frequency     = 2.0,
            .phaseShift    = 2.0
        }
    );

    _engine->getForceField()->addImproperDihedralType(dihedralType1);
    _engine->getForceField()->addImproperDihedralType(dihedralType2);

    auto setup = setup::ForceFieldSetup(*_engine);
    setup.setupImproperDihedrals();

    const auto &improperDihedrals =
        _engine->getForceField()->getImproperDihedrals();

    EXPECT_EQ(improperDihedrals[0].getType(), DihedralId{0});
    EXPECT_EQ(improperDihedrals[0].getParams().forceConstant, 1.0);
    EXPECT_EQ(improperDihedrals[0].getParams().phaseShift, 1.0);
    EXPECT_EQ(improperDihedrals[0].getParams().frequency, 1.0);

    EXPECT_EQ(improperDihedrals[1].getType(), DihedralId{1});
    EXPECT_EQ(improperDihedrals[1].getParams().forceConstant, 2.0);
    EXPECT_EQ(improperDihedrals[1].getParams().phaseShift, 2.0);
    EXPECT_EQ(improperDihedrals[1].getParams().frequency, 2.0);

    EXPECT_EQ(improperDihedrals[2].getType(), DihedralId{0});
    EXPECT_EQ(improperDihedrals[2].getParams().forceConstant, 1.0);
    EXPECT_EQ(improperDihedrals[2].getParams().phaseShift, 1.0);
    EXPECT_EQ(improperDihedrals[2].getParams().frequency, 1.0);

    EXPECT_EQ(_engine->getForceField()->getImproperTypes().size(), 0);
}

/**
 * @brief test setupForceField function
 *
 */
TEST_F(TestSetup, forceFieldSetupSetupForceField)
{
    settings::ForceFieldSettings::activate();

    auto molecule1 = molsys::Molecule();
    _engine->getSimulationBox().addMolecule(molecule1);
    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);

    auto bond = forceField::BondForceField(
        molecule1Ptr,
        molecule1Ptr,
        AtomIndex{0},
        AtomIndex{1},
        BondId{0}
    );
    auto angle = forceField::AngleForceField(
        {molecule1Ptr, molecule1Ptr, molecule1Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{0}
    );
    auto dihedral = forceField::DihedralForceField(
        {molecule1Ptr, molecule1Ptr, molecule1Ptr, molecule1Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );
    auto improperDihedral = forceField::DihedralForceField(
        {molecule1Ptr, molecule1Ptr, molecule1Ptr, molecule1Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );

    _engine->getForceField()->addBond(bond);
    _engine->getForceField()->addAngle(angle);
    _engine->getForceField()->addDihedral(dihedral);
    _engine->getForceField()->addImproperDihedral(improperDihedral);

    auto bondType = forceField::BondType(
        BondId{0},
        BondParams{.equilibrium = 1.0, .forceConstant = 2.0}
    );
    auto angleType = forceField::AngleType(
        AngleId{0},
        AngleParams{.equilibrium = 2.0, .forceConstant = 3.0}
    );
    auto dihedralType = forceField::DihedralType(
        DihedralId{0},
        DihedralParams{
            .forceConstant = 3.0,
            .frequency     = 4.0,
            .phaseShift    = 5.0
        }
    );
    auto improperDihedralType = forceField::DihedralType(
        DihedralId{0},
        DihedralParams{
            .forceConstant = 4.0,
            .frequency     = 5.0,
            .phaseShift    = 6.0
        }
    );

    _engine->getForceField()->addBondType(bondType);
    _engine->getForceField()->addAngleType(angleType);
    _engine->getForceField()->addDihedralType(dihedralType);
    _engine->getForceField()->addImproperDihedralType(improperDihedralType);

    setup::setupForceField(*_engine);

    const auto &bonds     = _engine->getForceField()->getBonds();
    const auto &angles    = _engine->getForceField()->getAngles();
    const auto &dihedrals = _engine->getForceField()->getDihedrals();
    const auto &improperDihedrals =
        _engine->getForceField()->getImproperDihedrals();

    EXPECT_EQ(bonds[0].getType(), BondId{0});
    EXPECT_EQ(bonds[0].getParams().equilibrium, 1.0);
    EXPECT_EQ(bonds[0].getParams().forceConstant, 2.0);

    EXPECT_EQ(angles[0].getType(), AngleId{0});
    EXPECT_EQ(angles[0].getParams().equilibrium, 2.0);
    EXPECT_EQ(angles[0].getParams().forceConstant, 3.0);

    EXPECT_EQ(dihedrals[0].getType(), DihedralId{0});
    EXPECT_EQ(dihedrals[0].getParams().forceConstant, 3.0);
    EXPECT_EQ(dihedrals[0].getParams().frequency, 4.0);
    EXPECT_EQ(dihedrals[0].getParams().phaseShift, 5.0);

    EXPECT_EQ(improperDihedrals[0].getType(), DihedralId{0});
    EXPECT_EQ(improperDihedrals[0].getParams().forceConstant, 4.0);
    EXPECT_EQ(improperDihedrals[0].getParams().frequency, 5.0);
    EXPECT_EQ(improperDihedrals[0].getParams().phaseShift, 6.0);
}

/**
 * @brief setupForceField should do nothing if force field is not activated
 *
 */
TEST_F(TestSetup, forceFieldSetupSetupForceFieldDoNothing)
{
    settings::ForceFieldSettings::activate();

    auto molecule1 = molsys::Molecule();
    _engine->getSimulationBox().addMolecule(molecule1);
    auto *molecule1Ptr = &_engine->getSimulationBox().getMolecule(0);

    auto bond = forceField::BondForceField(
        molecule1Ptr,
        molecule1Ptr,
        AtomIndex{0},
        AtomIndex{1},
        BondId{0}
    );
    auto angle = forceField::AngleForceField(
        {molecule1Ptr, molecule1Ptr, molecule1Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{0}
    );
    auto dihedral = forceField::DihedralForceField(
        {molecule1Ptr, molecule1Ptr, molecule1Ptr, molecule1Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );
    auto improperDihedral = forceField::DihedralForceField(
        {molecule1Ptr, molecule1Ptr, molecule1Ptr, molecule1Ptr},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );

    _engine->getForceField()->addBond(bond);
    _engine->getForceField()->addAngle(angle);
    _engine->getForceField()->addDihedral(dihedral);
    _engine->getForceField()->addImproperDihedral(improperDihedral);

    auto bondType = forceField::BondType(
        BondId{0},
        BondParams{.equilibrium = 1.0, .forceConstant = 2.0}
    );
    auto angleType = forceField::AngleType(
        AngleId{0},
        AngleParams{.equilibrium = 2.0, .forceConstant = 3.0}
    );
    auto dihedralType = forceField::DihedralType(
        DihedralId{0},
        DihedralParams{
            .forceConstant = 3.0,
            .frequency     = 4.0,
            .phaseShift    = 5.0
        }
    );
    auto improperDihedralType = forceField::DihedralType(
        DihedralId{0},
        DihedralParams{
            .forceConstant = 4.0,
            .frequency     = 5.0,
            .phaseShift    = 6.0
        }
    );

    _engine->getForceField()->addBondType(bondType);
    _engine->getForceField()->addAngleType(angleType);
    _engine->getForceField()->addDihedralType(dihedralType);
    _engine->getForceField()->addImproperDihedralType(improperDihedralType);

    settings::ForceFieldSettings::deactivate();
    setup::setupForceField(*_engine);

    const auto &bonds     = _engine->getForceField()->getBonds();
    const auto &angles    = _engine->getForceField()->getAngles();
    const auto &dihedrals = _engine->getForceField()->getDihedrals();
    const auto &improperDihedrals =
        _engine->getForceField()->getImproperDihedrals();

    EXPECT_NE(bonds[0].getParams().equilibrium, 1.0);
    EXPECT_NE(bonds[0].getParams().forceConstant, 2.0);

    EXPECT_NE(angles[0].getParams().equilibrium, 2.0);
    EXPECT_NE(angles[0].getParams().forceConstant, 3.0);

    EXPECT_NE(dihedrals[0].getParams().forceConstant, 3.0);
    EXPECT_NE(dihedrals[0].getParams().frequency, 4.0);
    EXPECT_NE(dihedrals[0].getParams().phaseShift, 5.0);

    EXPECT_NE(improperDihedrals[0].getParams().forceConstant, 4.0);
    EXPECT_NE(improperDihedrals[0].getParams().frequency, 5.0);
    EXPECT_NE(improperDihedrals[0].getParams().phaseShift, 6.0);
}
