#include "forceFieldSetup.hpp"
#include "testSetup.hpp"
#include "throwWithMessage.hpp"

#include <gtest/gtest.h>

/**
 * @brief test setupBonds function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupBonds)
{
    auto molecule1 = simulationBox::Molecule();
    auto molecule2 = simulationBox::Molecule();

    _engine.getSimulationBox().addMolecule(molecule1);
    _engine.getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine.getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine.getSimulationBox().getMolecule(1);

    auto bond1 = forceField::BondForceField(molecule1Ptr, molecule2Ptr, 0, 1, 0);
    auto bond2 = forceField::BondForceField(molecule1Ptr, molecule1Ptr, 0, 1, 1);
    auto bond3 = forceField::BondForceField(molecule1Ptr, molecule2Ptr, 0, 1, 0);

    _engine.getForceFieldPtr()->addBond(bond1);
    _engine.getForceFieldPtr()->addBond(bond2);
    _engine.getForceFieldPtr()->addBond(bond3);

    auto bondType1 = forceField::BondType(0, 1.0, 1.0);
    auto bondType2 = forceField::BondType(1, 2.0, 2.0);

    _engine.getForceFieldPtr()->addBondType(bondType1);
    _engine.getForceFieldPtr()->addBondType(bondType2);

    auto setup = setup::ForceFieldSetup(_engine);
    setup.setupBonds();

    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[0].getType(), 0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[0].getEquilibriumBondLength(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[0].getForceConstant(), 1.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[1].getType(), 1);
    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[1].getEquilibriumBondLength(), 2.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[1].getForceConstant(), 2.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[2].getType(), 0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[2].getEquilibriumBondLength(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getBonds()[2].getForceConstant(), 1.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getBondTypes().size(), 0);
}

/**
 * @brief test setupAngles function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupAngles)
{
    auto molecule1 = simulationBox::Molecule();
    auto molecule2 = simulationBox::Molecule();

    _engine.getSimulationBox().addMolecule(molecule1);
    _engine.getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine.getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine.getSimulationBox().getMolecule(1);

    auto angle1 = forceField::AngleForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2}, 0);
    auto angle2 = forceField::AngleForceField({molecule1Ptr, molecule1Ptr, molecule2Ptr}, {0, 1, 2}, 1);
    auto angle3 = forceField::AngleForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2}, 0);

    _engine.getForceFieldPtr()->addAngle(angle1);
    _engine.getForceFieldPtr()->addAngle(angle2);
    _engine.getForceFieldPtr()->addAngle(angle3);

    auto angleType1 = forceField::AngleType(0, 1.0, 1.0);
    auto angleType2 = forceField::AngleType(1, 2.0, 2.0);

    _engine.getForceFieldPtr()->addAngleType(angleType1);
    _engine.getForceFieldPtr()->addAngleType(angleType2);

    auto setup = setup::ForceFieldSetup(_engine);
    setup.setupAngles();

    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[0].getType(), 0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[0].getEquilibriumAngle(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[0].getForceConstant(), 1.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[1].getType(), 1);
    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[1].getEquilibriumAngle(), 2.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[1].getForceConstant(), 2.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[2].getType(), 0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[2].getEquilibriumAngle(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getAngles()[2].getForceConstant(), 1.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getAngleTypes().size(), 0);
}

/**
 * @brief test setupDihedrals function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupDihedrals)
{
    auto molecule1 = simulationBox::Molecule();
    auto molecule2 = simulationBox::Molecule();

    _engine.getSimulationBox().addMolecule(molecule1);
    _engine.getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine.getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine.getSimulationBox().getMolecule(1);

    auto dihedral1 = forceField::DihedralForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 0);
    auto dihedral2 = forceField::DihedralForceField({molecule1Ptr, molecule1Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 1);
    auto dihedral3 = forceField::DihedralForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 0);

    _engine.getForceFieldPtr()->addDihedral(dihedral1);
    _engine.getForceFieldPtr()->addDihedral(dihedral2);
    _engine.getForceFieldPtr()->addDihedral(dihedral3);

    auto dihedralType1 = forceField::DihedralType(0, 1.0, 1.0, 1.0);
    auto dihedralType2 = forceField::DihedralType(1, 2.0, 2.0, 2.0);

    _engine.getForceFieldPtr()->addDihedralType(dihedralType1);
    _engine.getForceFieldPtr()->addDihedralType(dihedralType2);

    auto setup = setup::ForceFieldSetup(_engine);
    setup.setupDihedrals();

    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[0].getType(), 0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[0].getForceConstant(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[0].getPhaseShift(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[0].getPeriodicity(), 1.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[1].getType(), 1);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[1].getForceConstant(), 2.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[1].getPhaseShift(), 2.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[1].getPeriodicity(), 2.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[2].getType(), 0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[2].getForceConstant(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[2].getPhaseShift(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedrals()[2].getPeriodicity(), 1.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getDihedralTypes().size(), 0);
}

/**
 * @brief test setupImproperDihedrals function
 *
 */
TEST_F(TestSetup, forceFieldSetup_setupImproperDihedrals)
{
    auto molecule1 = simulationBox::Molecule();
    auto molecule2 = simulationBox::Molecule();

    _engine.getSimulationBox().addMolecule(molecule1);
    _engine.getSimulationBox().addMolecule(molecule2);

    auto *molecule1Ptr = &_engine.getSimulationBox().getMolecule(0);
    auto *molecule2Ptr = &_engine.getSimulationBox().getMolecule(1);

    auto dihedral1 = forceField::DihedralForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 0);
    auto dihedral2 = forceField::DihedralForceField({molecule1Ptr, molecule1Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 1);
    auto dihedral3 = forceField::DihedralForceField({molecule1Ptr, molecule2Ptr, molecule2Ptr, molecule2Ptr}, {0, 1, 2, 3}, 0);

    _engine.getForceFieldPtr()->addImproperDihedral(dihedral1);
    _engine.getForceFieldPtr()->addImproperDihedral(dihedral2);
    _engine.getForceFieldPtr()->addImproperDihedral(dihedral3);

    auto dihedralType1 = forceField::DihedralType(0, 1.0, 1.0, 1.0);
    auto dihedralType2 = forceField::DihedralType(1, 2.0, 2.0, 2.0);

    _engine.getForceFieldPtr()->addImproperDihedralType(dihedralType1);
    _engine.getForceFieldPtr()->addImproperDihedralType(dihedralType2);

    auto setup = setup::ForceFieldSetup(_engine);
    setup.setupImproperDihedrals();

    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[0].getType(), 0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[0].getForceConstant(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[0].getPhaseShift(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[0].getPeriodicity(), 1.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[1].getType(), 1);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[1].getForceConstant(), 2.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[1].getPhaseShift(), 2.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[1].getPeriodicity(), 2.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[2].getType(), 0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[2].getForceConstant(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[2].getPhaseShift(), 1.0);
    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedrals()[2].getPeriodicity(), 1.0);

    EXPECT_EQ(_engine.getForceFieldPtr()->getImproperDihedralTypes().size(), 0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
