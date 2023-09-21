#include "atom.hpp"                      // for Atom
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "dihedralForceField.hpp"        // for BondForceField
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "matrix.hpp"                    // for Matrix
#include "molecule.hpp"                  // for Molecule
#include "physicalData.hpp"              // for PhysicalData
#include "potentialSettings.hpp"         // for PotentialSettings
#include "simulationBox.hpp"             // for SimulationBox
#include "vector3d.hpp"                  // for Vector3D, Vec3D, operator*

#include "gmock/gmock.h"   // for DoubleNear, ElementsAre
#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cmath>           // for sqrt
#include <cstddef>         // for size_t
#include <gtest/gtest.h>   // for EXPECT_NEAR, Test, InitGoogleTest, RUN_ALL_TESTS
#include <memory>          // for shared_ptr, allocator

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

TEST(TestDihedralForceField, calculateEnergyAndForces)
{
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData        = physicalData::PhysicalData();
    auto coulombPotential    = potential::CoulombShiftedPotential(20.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();

    auto nonCoulombPair = potential::LennardJonesPair(size_t(0), size_t(1), 15.0, 2.0, 4.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2));
    nonCoulombPotential.setNonCoulombPairsMatrix(0, 1, nonCoulombPair);

    auto molecule = simulationBox::Molecule();

    molecule.setMoltype(0);
    molecule.setNumberOfAtoms(4);

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();
    auto atom3 = std::make_shared<simulationBox::Atom>();
    auto atom4 = std::make_shared<simulationBox::Atom>();

    atom1->setPosition({0.0, 0.0, 0.0});
    atom2->setPosition({1.0, 1.0, 1.0});
    atom3->setPosition({1.0, 2.0, 3.0});
    atom4->setPosition({4.0, 2.0, 3.0});

    atom1->setForce({0.0, 0.0, 0.0});
    atom2->setForce({0.0, 0.0, 0.0});
    atom3->setForce({0.0, 0.0, 0.0});
    atom4->setForce({0.0, 0.0, 0.0});

    atom1->setInternalGlobalVDWType(0);
    atom2->setInternalGlobalVDWType(1);
    atom3->setInternalGlobalVDWType(0);
    atom4->setInternalGlobalVDWType(1);

    atom1->setAtomType(0);
    atom2->setAtomType(1);
    atom3->setAtomType(0);
    atom4->setAtomType(1);

    atom1->setPartialCharge(1.0);
    atom2->setPartialCharge(-0.5);
    atom3->setPartialCharge(1.0);
    atom4->setPartialCharge(-0.5);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);
    molecule.addAtom(atom4);

    auto bondForceField = forceField::DihedralForceField({&molecule, &molecule, &molecule, &molecule}, {0, 1, 2, 3}, 0);
    bondForceField.setPhaseShift(180.0 * M_PI / 180.0);
    bondForceField.setPeriodicity(3);
    bondForceField.setForceConstant(3.0);
    bondForceField.setIsLinker(false);

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.5);

    bondForceField.calculateEnergyAndForces(box, physicalData, false, coulombPotential, nonCoulombPotential);

    EXPECT_NEAR(physicalData.getDihedralEnergy(), 3.9128709291752739, 1e-6);
    EXPECT_NEAR(physicalData.getImproperEnergy(), 0.0, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], 3.19504825211347, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], -6.39009650422694, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], 3.19504825211347, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], -5.1120772033815518, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], 10.224154406763104, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -5.1120772033815518, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[0], 1.9170289512680818, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[1], -1.2780193008453877, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[2], 0.63900965042269386, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[0], 0.0, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[1], -2.5560386016907759, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[2], 1.278019300845388, 1e-6);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 0.0, 1e-6);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), 0.0, 1e-6);
    EXPECT_THAT(
        physicalData.getVirial(),
        testing::ElementsAre(testing::DoubleNear(0.0, 1e-6), testing::DoubleNear(0.0, 1e-6), testing::DoubleNear(0.0, 1e-6)));

    molecule.setAtomForce(0, {0.0, 0.0, 0.0});
    molecule.setAtomForce(1, {0.0, 0.0, 0.0});
    molecule.setAtomForce(2, {0.0, 0.0, 0.0});
    molecule.setAtomForce(3, {0.0, 0.0, 0.0});
    physicalData.reset();

    bondForceField.setIsLinker(false);

    bondForceField.calculateEnergyAndForces(box, physicalData, true, coulombPotential, nonCoulombPotential);

    EXPECT_NEAR(physicalData.getImproperEnergy(), 3.9128709291752739, 1e-6);
    EXPECT_NEAR(physicalData.getDihedralEnergy(), 0.0, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], 3.19504825211347, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], -6.39009650422694, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], 3.19504825211347, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], -5.1120772033815518, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], 10.224154406763104, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -5.1120772033815518, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[0], 1.9170289512680818, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[1], -1.2780193008453877, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[2], 0.63900965042269386, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[0], 0.0, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[1], -2.5560386016907759, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[2], 1.278019300845388, 1e-6);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 0.0, 1e-6);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), 0.0, 1e-6);
    EXPECT_THAT(
        physicalData.getVirial(),
        testing::ElementsAre(testing::DoubleNear(0.0, 1e-6), testing::DoubleNear(0.0, 1e-6), testing::DoubleNear(0.0, 1e-6)));

    molecule.setAtomForce(0, {0.0, 0.0, 0.0});
    molecule.setAtomForce(1, {0.0, 0.0, 0.0});
    molecule.setAtomForce(2, {0.0, 0.0, 0.0});
    molecule.setAtomForce(3, {0.0, 0.0, 0.0});
    physicalData.reset();

    bondForceField.setIsLinker(true);

    bondForceField.calculateEnergyAndForces(box, physicalData, false, coulombPotential, nonCoulombPotential);

    EXPECT_NEAR(physicalData.getDihedralEnergy(), 3.9128709291752739, 1e-6);
    EXPECT_NEAR(physicalData.getImproperEnergy(), 0.0, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], 2.2090108292824047, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], -6.8831152156424729, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], 2.4555201849901707, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], -5.1120772033815518, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], 10.224154406763104, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -5.1120772033815518, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[0], 1.9170289512680818, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[1], -1.2780193008453877, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(2)[2], 0.63900965042269386, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[0], 0.98603742283106555, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[1], -2.063019890275243, 1e-6);
    EXPECT_NEAR(molecule.getAtomForce(3)[2], 2.0175473679686871, 1e-6);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 4.1158570930777021, 1e-6);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), -4.100545344959669e-05, 1e-6);
    EXPECT_NEAR(physicalData.getVirial()[0], 3.9441496913242622, 1e-6);
    EXPECT_NEAR(physicalData.getVirial()[1], 0.98603742283106555, 1e-6);
    EXPECT_NEAR(physicalData.getVirial()[2], 2.2185842013698975, 1e-6);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}