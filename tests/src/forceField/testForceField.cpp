#include "angleForceField.hpp"           // for AngleForceField
#include "angleType.hpp"                 // for AngleType
#include "bondForceField.hpp"            // for BondForceField
#include "bondType.hpp"                  // for BondType
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "dihedralForceField.hpp"        // for DihedralForceField
#include "dihedralType.hpp"              // for DihedralType
#include "exceptions.hpp"                // for TopologyException
#include "forceField.hpp"                // for correctLinker
#include "forceFieldClass.hpp"           // for ForceField
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "matrix.hpp"                    // for Matrix
#include "molecule.hpp"                  // for Molecule
#include "physicalData.hpp"              // for PhysicalData
#include "potentialSettings.hpp"         // for PotentialSettings
#include "simulationBox.hpp"             // for SimulationBox
#include "throwWithMessage.hpp"          // for EXPECT_THROW_MSG
#include "vector3d.hpp"                  // for Vec3D

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cmath>           // for M_PI
#include <cstddef>         // for size_t
#include <gtest/gtest.h>   // for Test, CmpHelperNE, TestInfo
#include <memory>          // for shared_ptr, allocator
#include <string>          // for operator+, to_string, char_traits

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

/**
 * @brief tests findBondTypeById function
 *
 */
TEST(TestForceField, findBondTypeById)
{
    auto forceField = forceField::ForceField();
    auto bondType   = forceField::BondType(0, 1.0, 1.0);

    forceField.addBondType(bondType);

    EXPECT_EQ(forceField.findBondTypeById(0), bondType);
}

/**
 * @brief tests findBondTypeById function for not found error
 *
 */
TEST(TestForceField, findBondTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(forceField.findBondTypeById(0),
                     customException::TopologyException,
                     "Bond type with id " + std::to_string(0) + " not found.");
}

/**
 * @brief tests findAngleTypeById function
 *
 */
TEST(TestForceField, findAngleTypeById)
{
    auto forceField = forceField::ForceField();
    auto angleType  = forceField::AngleType(0, 1.0, 1.0);

    forceField.addAngleType(angleType);

    EXPECT_EQ(forceField.findAngleTypeById(0), angleType);
}

/**
 * @brief tests findAngleTypeById function for not found error
 *
 */
TEST(TestForceField, findAngleTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(forceField.findAngleTypeById(0),
                     customException::TopologyException,
                     "Angle type with id " + std::to_string(0) + " not found.");
}

/**
 * @brief tests findDihedralTypeById function
 *
 */
TEST(TestForceField, findDihedralTypeById)
{
    auto forceField   = forceField::ForceField();
    auto dihedralType = forceField::DihedralType(0, 1.0, 1.0, 1.0);

    forceField.addDihedralType(dihedralType);

    EXPECT_EQ(forceField.findDihedralTypeById(0), dihedralType);
}

/**
 * @brief tests findDihedralTypeById function for not found error
 *
 */
TEST(TestForceField, findDihedralTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(forceField.findDihedralTypeById(0),
                     customException::TopologyException,
                     "Dihedral type with id " + std::to_string(0) + " not found.");
}

/**
 * @brief tests findImproperDihedralTypeById function
 *
 */
TEST(TestForceField, findImproperDihedralTypeById)
{
    auto forceField           = forceField::ForceField();
    auto improperDihedralType = forceField::DihedralType(0, 1.0, 1.0, 1.0);

    forceField.addImproperDihedralType(improperDihedralType);

    EXPECT_EQ(forceField.findImproperDihedralTypeById(0), improperDihedralType);
}

/**
 * @brief tests findImproperDihedralTypeById function for not found error
 *
 */
TEST(TestForceField, findImproperDihedralTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(forceField.findImproperDihedralTypeById(0),
                     customException::TopologyException,
                     "Improper dihedral type with id " + std::to_string(0) + " not found.");
}

/**
 * @brief tests calculateBondedInteractions
 *
 * @details checks only if all energies are not zero - rest is checked in the respective test files
 *
 */
TEST(TestForceField, calculateBondedInteractions)
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
    molecule.setNumberOfAtoms(3);
    molecule.addAtomPosition({0.0, 0.0, 0.0});
    molecule.addAtomPosition({1.0, 1.0, 1.0});
    molecule.addAtomPosition({1.0, 2.0, 3.0});
    molecule.addAtomPosition({4.0, 2.0, 3.0});

    molecule.addAtomForce({0.0, 0.0, 0.0});
    molecule.addAtomForce({0.0, 0.0, 0.0});
    molecule.addAtomForce({0.0, 0.0, 0.0});
    molecule.addAtomForce({0.0, 0.0, 0.0});

    molecule.addInternalGlobalVDWType(0);
    molecule.addInternalGlobalVDWType(1);
    molecule.addInternalGlobalVDWType(0);
    molecule.addInternalGlobalVDWType(1);

    molecule.addAtomType(0);
    molecule.addAtomType(1);
    molecule.addAtomType(0);
    molecule.addAtomType(1);

    molecule.addPartialCharge(1.0);
    molecule.addPartialCharge(-0.5);
    molecule.addPartialCharge(1.0);
    molecule.addPartialCharge(-0.5);

    auto bondForceField     = forceField::BondForceField(&molecule, &molecule, 0, 1, 0);
    auto angleForceField    = forceField::AngleForceField({&molecule, &molecule, &molecule}, {0, 1, 2}, 0);
    auto dihedralForceField = forceField::DihedralForceField({&molecule, &molecule, &molecule, &molecule}, {0, 1, 2, 3}, 0);
    auto improperDihedralForceField =
        forceField::DihedralForceField({&molecule, &molecule, &molecule, &molecule}, {0, 1, 2, 3}, 0);

    bondForceField.setEquilibriumBondLength(1.2);
    bondForceField.setForceConstant(3.0);

    angleForceField.setEquilibriumAngle(90 * M_PI / 180.0);
    angleForceField.setForceConstant(3.0);

    dihedralForceField.setPhaseShift(180.0 * M_PI / 180.0);
    dihedralForceField.setPeriodicity(3);
    dihedralForceField.setForceConstant(3.0);
    dihedralForceField.setIsLinker(true);

    improperDihedralForceField.setPhaseShift(180.0 * M_PI / 180.0);
    improperDihedralForceField.setPeriodicity(3);
    improperDihedralForceField.setForceConstant(3.0);
    improperDihedralForceField.setIsLinker(false);

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.5);

    auto forceField = forceField::ForceField();

    forceField.addBond(bondForceField);
    forceField.addAngle(angleForceField);
    forceField.addDihedral(dihedralForceField);
    forceField.addImproperDihedral(improperDihedralForceField);
    forceField.setCoulombPotential(std::make_shared<potential::CoulombShiftedPotential>(coulombPotential));
    forceField.setNonCoulombPotential(std::make_shared<potential::ForceFieldNonCoulomb>(nonCoulombPotential));

    forceField.calculateBondedInteractions(box, physicalData);

    EXPECT_NE(physicalData.getBondEnergy(), 0.0);
    EXPECT_NE(physicalData.getAngleEnergy(), 0.0);
    EXPECT_NE(physicalData.getDihedralEnergy(), 0.0);
    EXPECT_NE(physicalData.getImproperEnergy(), 0.0);
    EXPECT_NE(physicalData.getCoulombEnergy(), 0.0);
    EXPECT_NE(physicalData.getNonCoulombEnergy(), 0.0);
    EXPECT_NE(physicalData.getVirial(), linearAlgebra::Vec3D(0.0, 0.0, 0.0));
}

/**
 * @brief test correctLinker
 *
 */
TEST(TestForceField, correctLinker)
{
    auto coulombPotential    = potential::CoulombShiftedPotential(10.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();

    auto nonCoulombPair = potential::LennardJonesPair(size_t(0), size_t(1), 5.0, 2.0, 4.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2));
    nonCoulombPotential.setNonCoulombPairsMatrix(0, 1, nonCoulombPair);

    auto molecule = simulationBox::Molecule();

    molecule.addAtomForce({0.0, 0.0, 0.0});
    molecule.addAtomForce({0.0, 0.0, 0.0});
    molecule.addInternalGlobalVDWType(0);
    molecule.addInternalGlobalVDWType(1);
    molecule.addAtomType(0);
    molecule.addAtomType(1);
    molecule.addPartialCharge(1.0);
    molecule.addPartialCharge(-0.5);

    physicalData::PhysicalData physicalData;

    const auto force =
        forceField::correctLinker(coulombPotential, nonCoulombPotential, physicalData, &molecule, &molecule, 0, 1, 1.0, false);

    EXPECT_NEAR(force, 104.37153798653807, 1e-9);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), -6, 1e-9);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 134.48580380716751, 1e-9);

    physicalData.clearData();

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.5);

    const auto forceScaled =
        forceField::correctLinker(coulombPotential, nonCoulombPotential, physicalData, &molecule, &molecule, 0, 1, 1.0, true);

    EXPECT_NEAR(forceScaled, 11.092884496634518, 1e-9);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), -3, 1e-9);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 33.621450951791878, 1e-9);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}