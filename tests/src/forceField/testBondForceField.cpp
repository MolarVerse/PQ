#include "atom.hpp"                      // for Atom
#include "bondForceField.hpp"            // for BondForceField
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

TEST(TestBondForceField, calculateEnergyAndForces)
{
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData        = physicalData::PhysicalData();
    auto coulombPotential    = potential::CoulombShiftedPotential(10.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();

    auto nonCoulombPair = potential::LennardJonesPair(size_t(0), size_t(1), 5.0, 2.0, 4.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2));
    nonCoulombPotential.setNonCoulombPairsMatrix(0, 1, nonCoulombPair);

    auto molecule = simulationBox::Molecule();

    molecule.setMoltype(0);
    molecule.setNumberOfAtoms(2);

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();

    atom1->setPosition({0.0, 0.0, 0.0});
    atom2->setPosition({1.0, 2.0, 3.0});
    atom1->setForce({0.0, 0.0, 0.0});
    atom2->setForce({0.0, 0.0, 0.0});
    atom1->setInternalGlobalVDWType(0);
    atom2->setInternalGlobalVDWType(1);
    atom1->setAtomType(0);
    atom2->setAtomType(1);
    atom1->setPartialCharge(1.0);
    atom2->setPartialCharge(-0.5);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);

    auto bondForceField = forceField::BondForceField(&molecule, &molecule, 0, 1, 0);
    bondForceField.setEquilibriumBondLength(1.2);
    bondForceField.setForceConstant(3.0);
    bondForceField.setIsLinker(false);

    bondForceField.calculateEnergyAndForces(box, physicalData, coulombPotential, nonCoulombPotential);

    auto force = (::sqrt(14) - 1.2) * 3.0 / ::sqrt(14) * linearAlgebra::Vec3D(1.0, 2.0, 3.0);

    EXPECT_NEAR(physicalData.getBondEnergy(), (::sqrt(14) - 1.2) * (::sqrt(14) - 1.2) * 3.0 / 2.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], force[0], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], force[1], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], force[2], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], -force[0], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], -force[1], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -force[2], 1e-9);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 0.0, 1e-9);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), 0.0, 1e-9);

    auto virial = force * linearAlgebra::Vec3D{-1.0, -2.0, -3.0};
    EXPECT_THAT(physicalData.getVirial(),
                testing::ElementsAre(testing::DoubleNear(virial[0], 1e-9),
                                     testing::DoubleNear(virial[1], 1e-9),
                                     testing::DoubleNear(virial[2], 1e-9)));

    molecule.setAtomForce(0, {0.0, 0.0, 0.0});
    molecule.setAtomForce(1, {0.0, 0.0, 0.0});
    physicalData.reset();

    bondForceField.setIsLinker(true);

    bondForceField.calculateEnergyAndForces(box, physicalData, coulombPotential, nonCoulombPotential);

    force = {-0.68765171383567736, -1.3753034276713547, -2.062955141507032};

    EXPECT_NEAR(physicalData.getBondEnergy(), (::sqrt(14) - 1.2) * (::sqrt(14) - 1.2) * 3.0 / 2.0, 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[0], force[0], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[1], force[1], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(0)[2], force[2], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[0], -force[0], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[1], -force[1], 1e-9);
    EXPECT_NEAR(molecule.getAtomForce(1)[2], -force[2], 1e-9);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 17.379852093794977, 1e-9);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), -0.00072939421499545261, 1e-9);

    virial = force * linearAlgebra::Vec3D{-1.0, -2.0, -3.0};
    EXPECT_THAT(physicalData.getVirial(),
                testing::ElementsAre(testing::DoubleNear(virial[0], 1e-9),
                                     testing::DoubleNear(virial[1], 1e-9),
                                     testing::DoubleNear(virial[2], 1e-9)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}