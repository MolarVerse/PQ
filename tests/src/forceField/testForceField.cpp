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

#include <memory>   // for shared_ptr, allocator

#include "../potential/nonCoulomb/testForceFieldNonCoulomb.hpp"
#include "angleForceField.hpp"           // for AngleForceField
#include "angleType.hpp"                 // for AngleType
#include "atom.hpp"                      // for Atom
#include "bondForceField.hpp"            // for BondForceField
#include "bondType.hpp"                  // for BondType
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "dihedralForceField.hpp"        // for DihedralForceField
#include "dihedralType.hpp"              // for DihedralType
#include "exceptions.hpp"                // for TopologyException
#include "forceField.hpp"             // IWYU pragma: keep - for correctLinker
#include "forceFieldClass.hpp"        // for ForceField
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "gtest/gtest.h"              // for Message, TestPartResult
#include "lennardJonesPair.hpp"       // for LennardJonesPair
#include "matrix.hpp"                 // for Matrix
#include "molecule.hpp"               // for Molecule
#include "physicalData.hpp"           // for PhysicalData
#include "potentialSettings.hpp"      // for PotentialSettings
#include "simulationBox.hpp"          // for SimulationBox
#include "strongTypes.hpp"
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

namespace potential
{
    class NonCoulombPair;   // forward declaration
}   // namespace potential

class TestForceField : public TestNonCoulombPotentialFF
{
};

/**
 * @brief tests findBondTypeById function
 *
 */
TEST_F(TestForceField, findBondTypeById)
{
    auto       forceField = forceField::ForceField();
    const auto bondType   = forceField::BondType(BondId{0}, 1.0, 1.0);

    forceField.addBondType(bondType);

    EXPECT_EQ(forceField.findBondTypeById(BondId{0}), bondType);
}

/**
 * @brief tests findBondTypeById function for not found error
 *
 */
TEST_F(TestForceField, findBondTypeByIdNotFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(
        const auto _ = forceField.findBondTypeById(BondId{0}),
        exc::TopologyException,
        "Bond type with id " + BondId(0).toString() + " not found."
    );
}

/**
 * @brief tests findAngleTypeById function
 *
 */
TEST_F(TestForceField, findAngleTypeById)
{
    auto forceField = forceField::ForceField();
    auto angleType  = forceField::AngleType(AngleId{0}, 1.0, 1.0);

    forceField.addAngleType(angleType);

    EXPECT_EQ(forceField.findAngleTypeById(AngleId{0}), angleType);
}

/**
 * @brief tests findAngleTypeById function for not found error
 *
 */
TEST_F(TestForceField, findAngleTypeByIdNotFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(
        const auto _ = forceField.findAngleTypeById(AngleId{0}),
        exc::TopologyException,
        "Angle type with id " + AngleId(0).toString() + " not found."
    );
}

/**
 * @brief tests findDihedralTypeById function
 *
 */
TEST_F(TestForceField, findDihedralTypeById)
{
    auto forceField   = forceField::ForceField();
    auto dihedralType = forceField::DihedralType(DihedralId{0}, 1.0, 1.0, 1.0);

    forceField.addDihedralType(dihedralType);

    EXPECT_EQ(forceField.findDihedralTypeById(DihedralId{0}), dihedralType);
}

/**
 * @brief tests findDihedralTypeById function for not found error
 *
 */
TEST_F(TestForceField, findDihedralTypeByIdNotFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(
        const auto _ = forceField.findDihedralTypeById(DihedralId{0}),
        exc::TopologyException,
        "Dihedral type with id " + DihedralId(0).toString() + " not found."
    );
}

/**
 * @brief tests findImproperTypeById function
 *
 */
TEST_F(TestForceField, findImproperTypeById)
{
    auto forceField = forceField::ForceField();
    auto improperDihedralType =
        forceField::DihedralType(DihedralId{0}, 1.0, 1.0, 1.0);

    forceField.addImproperDihedralType(improperDihedralType);

    EXPECT_EQ(
        forceField.findImproperTypeById(DihedralId{0}),
        improperDihedralType
    );
}

/**
 * @brief tests findImproperTypeById function for not found error
 *
 */
TEST_F(TestForceField, findImproperDihedralTypeByIdNotFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(
        const auto _ = forceField.findImproperTypeById(DihedralId{0}),
        exc::TopologyException,
        "Improper dihedral type with id " + DihedralId(0).toString() +
            " not found."
    );
}

/**
 * @brief tests calculateBondedInteractions
 *
 * @details checks only if all energies are not zero - rest is checked in the
 * respective test files
 *
 */
TEST_F(TestForceField, calculateBondedInteractions)
{
    auto box = molsys::SimulationBox();
    box.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData     = physicalData::PhysicalData();
    auto coulombPotential = potential::CoulombShiftedPotential(20.0);

    auto nonCoulombPair = potential::LennardJonesPair(
        ExtVdwType(0),
        ExtVdwType(1),
        15.0,
        LJParams{.c6 = 2.0, .c12 = 4.0}
    );
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2)
    );
    setNonCoulombPairsMatrix(0, 1, nonCoulombPair);

    auto molecule = molsys::Molecule();

    molecule.setMoltype(0);
    molecule.setNumberOfAtoms(4);

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();
    auto atom3 = std::make_shared<molsys::Atom>();
    auto atom4 = std::make_shared<molsys::Atom>();

    atom1->setPosition({0.0, 0.0, 0.0});
    atom2->setPosition({1.0, 1.0, 1.0});
    atom3->setPosition({1.0, 2.0, 3.0});
    atom4->setPosition({4.0, 2.0, 3.0});

    atom1->setForce({0.0, 0.0, 0.0});
    atom2->setForce({0.0, 0.0, 0.0});
    atom3->setForce({0.0, 0.0, 0.0});
    atom4->setForce({0.0, 0.0, 0.0});

    atom1->setInternalGlobalVDWType(VdwType{0});
    atom2->setInternalGlobalVDWType(VdwType{1});
    atom3->setInternalGlobalVDWType(VdwType{0});
    atom4->setInternalGlobalVDWType(VdwType{1});

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

    auto bondForceField = forceField::BondForceField(
        &molecule,
        &molecule,
        AtomIndex{0},
        AtomIndex{1},
        BondId{0}
    );
    auto angleForceField = forceField::AngleForceField(
        {&molecule, &molecule, &molecule},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{0}
    );
    auto dihedralForceField = forceField::DihedralForceField(
        {&molecule, &molecule, &molecule, &molecule},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );
    auto improperDihedralForceField = forceField::DihedralForceField(
        {&molecule, &molecule, &molecule, &molecule},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
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
        std::make_shared<potential::ForceFieldNonCoulomb>(*_nonCoulombPotential)
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
TEST_F(TestForceField, correctLinker)
{
    auto coulombPotential = potential::CoulombShiftedPotential(10.0);

    auto nonCoulombPair = potential::LennardJonesPair(
        ExtVdwType(0),
        ExtVdwType(1),
        5.0,
        LJParams{.c6 = 2.0, .c12 = 4.0}
    );
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(2, 2)
    );
    setNonCoulombPairsMatrix(0, 1, nonCoulombPair);

    auto molecule = molsys::Molecule();

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();

    atom1->setForce({0.0, 0.0, 0.0});
    atom2->setForce({0.0, 0.0, 0.0});
    atom1->setInternalGlobalVDWType(VdwType{0});
    atom2->setInternalGlobalVDWType(VdwType{1});
    atom1->setAtomType(0);
    atom2->setAtomType(1);
    atom1->setPartialCharge(1.0);
    atom2->setPartialCharge(-0.5);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);

    physicalData::PhysicalData physicalData;

    const auto force = forceField::correctLinker<forceField::BondForceField>(
        coulombPotential,
        *_nonCoulombPotential,
        physicalData,
        &molecule,
        &molecule,
        AtomIndex{0},
        AtomIndex{1},
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
            *_nonCoulombPotential,
            physicalData,
            &molecule,
            &molecule,
            AtomIndex{0},
            AtomIndex{1},
            1.0
        );

    EXPECT_NEAR(forceScaled, 11.092884496634518, 1e-6);
    EXPECT_NEAR(physicalData.getNonCoulombEnergy(), -3, 1e-6);
    EXPECT_NEAR(physicalData.getCoulombEnergy(), 33.621450951791878, 1e-6);
}
