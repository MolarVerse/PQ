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

#include <gtest/gtest.h>   // for Test, CmpHelperNE, TestInfo

#include <cmath>     // for M_PI
#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr, allocator
#include <string>    // for operator+, to_string, char_traits

#include "angleForceField.hpp"           // for AngleForceField
#include "angleType.hpp"                 // for AngleType
#include "atom.hpp"                      // for Atom
#include "bondForceField.hpp"            // for BondForceField
#include "bondType.hpp"                  // for BondType
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "dihedralForceField.hpp"        // for DihedralForceField
#include "dihedralType.hpp"              // for DihedralType
#include "exceptions.hpp"                // for TopologyException
#include "forceField.hpp"                // for correctLinker
#include "forceFieldClass.hpp"           // for ForceField
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "gtest/gtest.h"                 // for Message, TestPartResult
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "matrix.hpp"                    // for Matrix
#include "molecule.hpp"                  // for Molecule
#include "physicalData.hpp"              // for PhysicalData
#include "potentialSettings.hpp"         // for PotentialSettings
#include "simulationBox.hpp"             // for SimulationBox
#include "throwWithMessage.hpp"          // for EXPECT_THROW_MSG
#include "vector3d.hpp"                  // for Vec3D

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
    auto       forceField = forceField::ForceField();
    const auto bondType   = forceField::BondType(0, 1.0, 1.0);

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

    EXPECT_THROW_MSG(
        forceField.findBondTypeById(0),
        customException::TopologyException,
        "Bond type with id " + std::to_string(0) + " not found."
    );
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

    EXPECT_THROW_MSG(
        forceField.findAngleTypeById(0),
        customException::TopologyException,
        "Angle type with id " + std::to_string(0) + " not found."
    );
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

    EXPECT_THROW_MSG(
        forceField.findDihedralTypeById(0),
        customException::TopologyException,
        "Dihedral type with id " + std::to_string(0) + " not found."
    );
}

/**
 * @brief tests findImproperTypeById function
 *
 */
TEST(TestForceField, findImproperTypeById)
{
    auto forceField           = forceField::ForceField();
    auto improperDihedralType = forceField::DihedralType(0, 1.0, 1.0, 1.0);

    forceField.addImproperDihedralType(improperDihedralType);

    EXPECT_EQ(forceField.findImproperTypeById(0), improperDihedralType);
}

/**
 * @brief tests findImproperTypeById function for not found error
 *
 */
TEST(TestForceField, findImproperDihedralTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(
        forceField.findImproperTypeById(0),
        customException::TopologyException,
        "Improper dihedral type with id " + std::to_string(0) + " not found."
    );
}

/**
 * @brief tests calculateBondedInteractions
 *
 * @details checks only if all energies are not zero - rest is checked in the
 * respective test files
 *
 */
TEST(TestForceField, calculateBondedInteractions)
{
    auto box = simulationBox::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData        = physicalData::PhysicalData();
    auto coulombPotential    = potential::CoulombShiftedPotential(20.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();

    auto nonCoulombPair =
        potential::LennardJonesPair(size_t(0), size_t(1), 15.0, 2.0, 4.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2)
    );
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

    auto bondForceField =
        forceField::BondForceField(&molecule, &molecule, 0, 1, 0);
    auto angleForceField = forceField::AngleForceField(
        {&molecule, &molecule, &molecule},
        {0, 1, 2},
        0
    );
    auto dihedralForceField = forceField::DihedralForceField(
        {&molecule, &molecule, &molecule, &molecule},
        {0, 1, 2, 3},
        0
    );
    auto improperDihedralForceField = forceField::DihedralForceField(
        {&molecule, &molecule, &molecule, &molecule},
        {0, 1, 2, 3},
        0
    );

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
    forceField.setCoulombPotential(
        std::make_shared<potential::CoulombShiftedPotential>(coulombPotential)
    );
    forceField.setNonCoulombPotential(
        std::make_shared<potential::ForceFieldNonCoulomb>(nonCoulombPotential)
    );

    forceField.calculateBondedInteractions(box, physicalData);

    EXPECT_NE(physicalData.getBondEnergy(), 0.0);
    EXPECT_NE(physicalData.getAngleEnergy(), 0.0);
    EXPECT_NE(physicalData.getDihedralEnergy(), 0.0);
    EXPECT_NE(physicalData.getImproperEnergy(), 0.0);
    EXPECT_NE(physicalData.getCoulombEnergy(), 0.0);
    EXPECT_NE(physicalData.getNonCoulombEnergy(), 0.0);
    EXPECT_NE(physicalData.getVirial(), linearAlgebra::tensor3D(0.0));
}

/**
 * @brief test correctLinker
 *
 */
TEST(TestForceField, correctLinker)
{
    auto coulombPotential    = potential::CoulombShiftedPotential(10.0);
    auto nonCoulombPotential = potential::ForceFieldNonCoulomb();

    auto nonCoulombPair =
        potential::LennardJonesPair(size_t(0), size_t(1), 5.0, 2.0, 4.0);
    nonCoulombPotential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2)
    );
    nonCoulombPotential.setNonCoulombPairsMatrix(0, 1, nonCoulombPair);

    auto molecule = simulationBox::Molecule();

    auto atom1 = std::make_shared<simulationBox::Atom>();
    auto atom2 = std::make_shared<simulationBox::Atom>();

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

    physicalData::PhysicalData physicalData;

    const auto force = forceField::correctLinker<forceField::BondForceField>(
        coulombPotential,
        nonCoulombPotential,
        physicalData,
        &molecule,
        &molecule,
        0,
        1,
        1.0
    );

    EXPECT_NEAR(force, 104.37153798653807, 1e-6);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), -6, 1e-6);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 134.48580380716751, 1e-6);

    physicalData.reset();

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.5);

    const auto forceScaled =
        forceField::correctLinker<forceField::DihedralForceField>(
            coulombPotential,
            nonCoulombPotential,
            physicalData,
            &molecule,
            &molecule,
            0,
            1,
            1.0
        );

    EXPECT_NEAR(forceScaled, 11.092884496634518, 1e-6);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), -3, 1e-6);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 33.621450951791878, 1e-6);
}