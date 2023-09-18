#include "angleForceField.hpp"           // for BondForceField
#include "atom.hpp"                      // for Atom
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "matrix.hpp"                    // for Matrix
#include "molecule.hpp"                  // for Molecule
#include "physicalData.hpp"              // for PhysicalData
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

TEST(TestAngleForceField, calculateEnergyAndForces)
{
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData        = physicalData::PhysicalData();
    auto coulombPotential    = potential::CoulombShiftedPotential(10.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();

    auto nonCoulombPair = potential::LennardJonesPair(size_t(1), size_t(1), 5.0, 2.0, 4.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2));
    nonCoulombPotential.setNonCoulombPairsMatrix(1, 1, nonCoulombPair);

    auto molecule = simulationBox::Molecule();

    molecule.setMoltype(0);
    molecule.setNumberOfAtoms(3);

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();
    auto atom3 = std::make_shared<simulationBox::Atom>();

    atom1->setPosition({0.0, 0.0, 0.0});
    atom2->setPosition({1.0, 1.0, 1.0});
    atom3->setPosition({1.0, 2.0, 3.0});

    atom1->setForce({0.0, 0.0, 0.0});
    atom2->setForce({0.0, 0.0, 0.0});
    atom3->setForce({0.0, 0.0, 0.0});

    atom1->setInternalGlobalVDWType(0);
    atom2->setInternalGlobalVDWType(1);
    atom3->setInternalGlobalVDWType(1);

    atom1->setAtomType(0);
    atom2->setAtomType(1);
    atom3->setAtomType(1);

    atom1->setPartialCharge(1.0);
    atom2->setPartialCharge(-0.5);
    atom3->setPartialCharge(-0.5);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    auto bondForceField = forceField::AngleForceField({&molecule, &molecule, &molecule}, {0, 1, 2}, 0);
    bondForceField.setEquilibriumAngle(90 * M_PI / 180.0);
    bondForceField.setForceConstant(3.0);
    bondForceField.setIsLinker(false);

    bondForceField.calculateEnergyAndForces(box, physicalData, coulombPotential, nonCoulombPotential);

    EXPECT_NEAR(physicalData.getAngleEnergy(), 2.0999420826401303, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], -0.62105043904006785, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], 0.20701681301335595, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], 1.0350840650667796, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], 1.4491176910934915, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -1.4491176910934915, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(2)[0], -0.82806725205342369, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(2)[1], -0.20701681301335595, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(2)[2], 0.41403362602671184, 1e-9);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 0.0, 1e-9);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), 0.0, 1e-9);
    EXPECT_THAT(
        physicalData.getVirial(),
        testing::ElementsAre(testing::DoubleNear(0.0, 1e-9), testing::DoubleNear(0.0, 1e-9), testing::DoubleNear(0.0, 1e-9)));

    molecule.setAtomForce(0, {0.0, 0.0, 0.0});
    molecule.setAtomForce(1, {0.0, 0.0, 0.0});
    molecule.setAtomForce(2, {0.0, 0.0, 0.0});
    physicalData.reset();

    bondForceField.setIsLinker(true);

    bondForceField.calculateEnergyAndForces(box, physicalData, coulombPotential, nonCoulombPotential);

    EXPECT_NEAR(physicalData.getAngleEnergy(), 2.0999420826401303, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], -0.62105043904006785, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], 0.20701681301335595, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], 1.0350840650667796, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], 1.4491176910934915, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], 7.0737262359370403, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], 12.69833478078059, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(2)[0], -0.82806725205342369, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(2)[1], -7.2807430489503959, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(2)[2], -13.733418845847369, 1e-9);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), -22.378958701288319, 1e-9);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), -0.016255999999999989, 1e-9);
    EXPECT_NEAR(physicalData.getVirial()[0], 0.0, 1e-9);
    EXPECT_NEAR(physicalData.getVirial()[1], -7.0737262359370403, 1e-9);
    EXPECT_NEAR(physicalData.getVirial()[2], -28.294904943748161, 1e-9);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}