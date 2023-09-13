#include "atom.hpp"                      // for Atom
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "intraNonBondedMap.hpp"         // for IntraNonBondedMap
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "matrix.hpp"                    // for Matrix
#include "molecule.hpp"                  // for Molecule
#include "physicalData.hpp"              // for PhysicalData
#include "potentialSettings.hpp"         // for PotentialSettings
#include "simulationBox.hpp"             // for SimulationBox
#include "vector3d.hpp"                  // for Vec3D

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cstddef>         // for size_t
#include <gtest/gtest.h>   // for Test, EXPECT_NEAR, InitGoogleTest, RUN_ALL.
#include <memory>          // for shared_ptr, allocator
#include <vector>          // for vector

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

/**
 * @brief Test fixture class for the IntraNonBondedMap class
 */
TEST(testIntraNonBondedMap, calculateSingleInteraction_AND_calculate)
{
    auto molecule = simulationBox::Molecule(0);
    molecule.setNumberOfAtoms(2);

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();

    atom1->setPosition({0.0, 0.0, 0.0});
    atom2->setPosition({0.0, 0.0, 11.0});
    atom1->setForce({0.0, 0.0, 0.0});
    atom2->setForce({0.0, 0.0, 0.0});
    atom1->setInternalGlobalVDWType(0);
    atom2->setInternalGlobalVDWType(1);
    atom1->setAtomType(0);
    atom2->setAtomType(1);
    atom1->setPartialCharge(0.5);
    atom2->setPartialCharge(-0.5);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.75);

    auto intraNonBondedType = intraNonBonded::IntraNonBondedContainer(0, {{-1}});
    auto intraNonBondedMap  = intraNonBonded::IntraNonBondedMap(&molecule, &intraNonBondedType);

    auto coulombPotential    = potential::CoulombShiftedPotential(10.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();
    nonCoulombPotential.setNonCoulombPairsMatrix(linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2));

    auto nonCoulombPair = potential::LennardJonesPair(size_t(0), size_t(1), 10.0, 2.0, 3.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(0, 1, nonCoulombPair);
    nonCoulombPotential.setNonCoulombPairsMatrix(1, 0, nonCoulombPair);

    auto simulationBox = simulationBox::SimulationBox();
    simulationBox.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData = physicalData::PhysicalData();

    const auto [coulombEnergy, nonCoulombEnergy] =
        intraNonBondedMap.calculateSingleInteraction(0,
                                                     intraNonBondedType.getAtomIndices()[0][0],
                                                     simulationBox.getBoxDimensions(),
                                                     physicalData,
                                                     &coulombPotential,
                                                     &nonCoulombPotential);

    EXPECT_NEAR(coulombEnergy, -67.242901903583757 * 0.75, 1e-9);
    EXPECT_NEAR(nonCoulombEnergy, 5.0 * 0.75, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], 34.185768993269036 * 0.75, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -34.185768993269036 * 0.75, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[2], 341.85768993269039 * 0.75, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[2], 0.0, 1e-9);

    molecule.setAtomForcesToZero();
    molecule.getAtom(0).setShiftForce({0.0, 0.0, 0.0});
    molecule.getAtom(1).setShiftForce({0.0, 0.0, 0.0});
    physicalData.reset();

    intraNonBondedMap.calculate(&coulombPotential, &nonCoulombPotential, simulationBox, physicalData);

    EXPECT_NEAR(physicalData.getCoulombEnergy(), -67.242901903583757 * 0.75, 1e-9);
    EXPECT_NEAR(physicalData.getIntraCoulombEnergy(), -67.242901903583757 * 0.75, 1e-9);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), 5.0 * 0.75, 1e-9);
    EXPECT_NEAR(physicalData.getIntraNonCoulombEnergy(), 5.0 * 0.75, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], 34.185768993269036 * 0.75, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -34.185768993269036 * 0.75, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[2], 341.85768993269039 * 0.75, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[2], 0.0, 1e-9);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}