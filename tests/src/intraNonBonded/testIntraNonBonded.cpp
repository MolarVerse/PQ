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

#include "atom.hpp"                      // for Atom
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "exceptions.hpp"                // for IntraNonBondedException
#include "forceFieldNonCoulomb.hpp"      // for ForceFieldNonCoulomb
#include "intraNonBonded.hpp"            // for IntraNonBonded
#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "intraNonBondedMap.hpp"         // for IntraNonBondedMap
#include "lennardJonesPair.hpp"          // for LennardJonesPair
#include "matrix.hpp"                    // for Matrix
#include "molecule.hpp"                  // for Molecule
#include "physicalData.hpp"              // for PhysicalData
#include "potentialSettings.hpp"         // for PotentialSettings
#include "simulationBox.hpp"             // for SimulationBox
#include "throwWithMessage.hpp"          // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <cstddef>         // for size_t
#include <format>          // for format
#include <gtest/gtest.h>   // for Test, EXPECT_EQ, TestInfo
#include <memory>          // for shared_ptr, allocator, make_shared
#include <string>          // for basic_string
#include <vector>          // for vector

namespace potential
{
    class NonCoulombPair;   // forward declaration
}

/**
 * @brief test findIntraNonBondedContainerByMolType method
 */
TEST(testIntraNonBonded, findIntraNonBondedContainerByMolType)
{
    const auto intraNonBondedContainer1 = intraNonBonded::IntraNonBondedContainer(0, {{-1}});
    const auto intraNonBondedContainer2 = intraNonBonded::IntraNonBondedContainer(1, {{-1}});
    const auto intraNonBondedContainer3 = intraNonBonded::IntraNonBondedContainer(2, {{-1}});

    auto intraNonBonded = intraNonBonded::IntraNonBonded();
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer1);
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer2);
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer3);

    const auto *intraNonBondedContainerPtr = intraNonBonded.findIntraNonBondedContainerByMolType(1);

    EXPECT_EQ(intraNonBondedContainerPtr->getMolType(), intraNonBondedContainer2.getMolType());
    EXPECT_EQ(intraNonBondedContainerPtr->getAtomIndices(), intraNonBondedContainer2.getAtomIndices());

    EXPECT_THROW_MSG([[maybe_unused]] const auto dummy = intraNonBonded.findIntraNonBondedContainerByMolType(3),
                     customException::IntraNonBondedException,
                     std::format("IntraNonBondedContainer with molType 3 not found!"))
}

/**
 * @brief test fillIntraNonBondedMaps method
 */
TEST(testIntraNonBonded, fillIntraNonBondedMaps)
{
    const auto intraNonBondedContainer1 = intraNonBonded::IntraNonBondedContainer(0, {{-1}});
    const auto intraNonBondedContainer2 = intraNonBonded::IntraNonBondedContainer(1, {{-1}});
    const auto intraNonBondedContainer3 = intraNonBonded::IntraNonBondedContainer(2, {{-1}});

    auto intraNonBonded = intraNonBonded::IntraNonBonded();
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer1);
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer2);
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer3);

    auto simulationBox = simulationBox::SimulationBox();
    auto molecule1     = simulationBox::Molecule(0);
    auto molecule2     = simulationBox::Molecule(1);
    auto molecule3     = simulationBox::Molecule(2);
    auto molecule4     = simulationBox::Molecule(1);
    auto molecule5     = simulationBox::Molecule(2);

    simulationBox.addMolecule(molecule1);
    simulationBox.addMolecule(molecule2);
    simulationBox.addMolecule(molecule3);
    simulationBox.addMolecule(molecule4);
    simulationBox.addMolecule(molecule5);

    intraNonBonded.fillIntraNonBondedMaps(simulationBox);

    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps().size(), 5);
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[0].getMolecule(), &simulationBox.getMolecule(0));
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[0].getAtomIndices(), intraNonBondedContainer1.getAtomIndices());
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[1].getMolecule(), &simulationBox.getMolecule(1));
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[1].getAtomIndices(), intraNonBondedContainer2.getAtomIndices());
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[2].getMolecule(), &simulationBox.getMolecule(2));
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[2].getAtomIndices(), intraNonBondedContainer3.getAtomIndices());
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[3].getMolecule(), &simulationBox.getMolecule(3));
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[3].getAtomIndices(), intraNonBondedContainer2.getAtomIndices());
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[4].getMolecule(), &simulationBox.getMolecule(4));
    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps()[4].getAtomIndices(), intraNonBondedContainer3.getAtomIndices());
}

/**
 * @brief test calculate method
 *
 * @details only wrapper for calculate method of IntraNonBondedMap class
 */
TEST(TestIntraNonBonded, calculate)
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

    auto intraNonBonded = intraNonBonded::IntraNonBonded();
    intraNonBonded.addIntraNonBondedMap(intraNonBondedMap);

    intraNonBonded.setCoulombPotential(std::make_shared<potential::CoulombShiftedPotential>(coulombPotential));
    intraNonBonded.setNonCoulombPotential(std::make_shared<potential::ForceFieldNonCoulomb>(nonCoulombPotential));
    EXPECT_NO_THROW(intraNonBonded.calculate(simulationBox, physicalData));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}