#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "intraNonBondedMap.hpp"         // for IntraNonBondedMap
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "matrix.hpp"                    // for Matrix
#include "molecule.hpp"                  // for Molecule
#include "simulationBox.hpp"             // for SimulationBox
#include "vector3d.hpp"                  // for Vec3D

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cstddef>         // for size_t
#include <gtest/gtest.h>   // for Test, EXPECT_NEAR, InitGoogle...
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
    molecule.addAtomPosition({0.0, 0.0, 0.0});
    molecule.addAtomPosition({0.0, 0.0, 11.0});
    molecule.addAtomForce({0.0, 0.0, 0.0});
    molecule.addAtomForce({0.0, 0.0, 0.0});
    molecule.addInternalGlobalVDWType(0);
    molecule.addInternalGlobalVDWType(1);
    molecule.addAtomType(0);
    molecule.addAtomType(1);
    molecule.addPartialCharge(0.5);
    molecule.addPartialCharge(-0.5);
    molecule.resizeAtomShiftForces();

    auto intraNonBondedType = intraNonBonded::IntraNonBondedContainer(0, {{1}});
    auto intraNonBondedMap  = intraNonBonded::IntraNonBondedMap(&molecule, &intraNonBondedType);

    auto coulombPotential    = potential::CoulombShiftedPotential(10.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();
    nonCoulombPotential.setNonCoulombPairsMatrix(linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2));

    auto nonCoulombPair = potential::LennardJonesPair(size_t(0), size_t(1), 10.0, 2.0, 3.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(0, 1, nonCoulombPair);
    nonCoulombPotential.setNonCoulombPairsMatrix(1, 0, nonCoulombPair);

    auto simulationBox = simulationBox::SimulationBox();
    simulationBox.setBoxDimensions({10.0, 10.0, 10.0});

    const auto [coulombEnergy, nonCoulombEnergy] = intraNonBondedMap.calculateSingleInteraction(
        0, intraNonBondedType.getAtomIndices()[0][0], simulationBox.getBoxDimensions(), &coulombPotential, &nonCoulombPotential);

    EXPECT_NEAR(coulombEnergy, -67.242901903583757, 1e-9);
    EXPECT_NEAR(nonCoulombEnergy, 5.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], 34.185768993269036, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -34.185768993269036, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(0)[2], 341.85768993269039, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[0], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[1], 0.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomShiftForce(1)[2], 0.0, 1e-9);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}