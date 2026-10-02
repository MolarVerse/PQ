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

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "../potential/nonCoulomb/testForceFieldNonCoulomb.hpp"
#include "atom.hpp"
#include "coulombShiftedPotential.hpp"
#include "exceptions.hpp"
#include "forceFieldNonCoulomb.hpp"
                                         // for Message, TestPartResult
#include "intraNonBonded.hpp"
#include "intraNonBondedContainer.hpp"
#include "intraNonBondedMap.hpp"
#include "lennardJonesPair.hpp"
#include "matrix.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "throwWithMessage.hpp"

namespace pot
{
    class NonCoulombPair;   // forward declaration
}   // namespace pot

class TestIntraNonBonded : public TestNonCoulombPotentialFF
{
};

/**
 * @brief test findIntraNonBondedContainerByMolType method
 */
TEST_F(TestIntraNonBonded, findIntraNonBondedContainerByMolType)
{
    const auto intraNonBondedContainer1 =
        intraNonBonded::IntraNonBondedContainer(MolType{0}, {{-1}});
    const auto intraNonBondedContainer2 =
        intraNonBonded::IntraNonBondedContainer(MolType{1}, {{-1}});
    const auto intraNonBondedContainer3 =
        intraNonBonded::IntraNonBondedContainer(MolType{2}, {{-1}});

    auto intraNonBonded = intraNonBonded::IntraNonBonded();
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer1);
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer2);
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer3);

    const auto *intraNonBondedContainerPtr =
        intraNonBonded.findIntraNonBondedContainerByMolType(MolType{1});

    EXPECT_EQ(
        intraNonBondedContainerPtr->getMolType(),
        intraNonBondedContainer2.getMolType()
    );
    EXPECT_EQ(
        intraNonBondedContainerPtr->getAtomIndices(),
        intraNonBondedContainer2.getAtomIndices()
    );

    EXPECT_THROW_MSG(
        [[maybe_unused]] const auto dummy =
            intraNonBonded.findIntraNonBondedContainerByMolType(MolType{3}),
        exc::IntraNonBondedException,
        "IntraNonBondedContainer with molType MolType(3) not found!"
    )
}

/**
 * @brief test fillIntraNonBondedMaps method
 */
TEST_F(TestIntraNonBonded, fillIntraNonBondedMaps)
{
    const auto intraNonBondedContainer1 =
        intraNonBonded::IntraNonBondedContainer(MolType{0}, {{-1}});
    const auto intraNonBondedContainer2 =
        intraNonBonded::IntraNonBondedContainer(MolType{1}, {{-1}});
    const auto intraNonBondedContainer3 =
        intraNonBonded::IntraNonBondedContainer(MolType{2}, {{-1}});

    auto intraNonBonded = intraNonBonded::IntraNonBonded();
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer1);
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer2);
    intraNonBonded.addIntraNonBondedContainer(intraNonBondedContainer3);

    auto simulationBox = molsys::SimulationBox();
    auto molecule1     = molsys::Molecule{MolType{0}};
    auto molecule2     = molsys::Molecule{MolType{1}};
    auto molecule3     = molsys::Molecule{MolType{2}};
    auto molecule4     = molsys::Molecule{MolType{1}};
    auto molecule5     = molsys::Molecule{MolType{2}};

    simulationBox.addMolecule(molecule1);
    simulationBox.addMolecule(molecule2);
    simulationBox.addMolecule(molecule3);
    simulationBox.addMolecule(molecule4);
    simulationBox.addMolecule(molecule5);

    intraNonBonded.fillIntraNonBondedMaps(simulationBox);

    EXPECT_EQ(intraNonBonded.getIntraNonBondedMaps().size(), 5);
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[0].getMolecule(),
        &simulationBox.getMolecule(0)
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[0].getAtomIndices(),
        intraNonBondedContainer1.getAtomIndices()
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[1].getMolecule(),
        &simulationBox.getMolecule(1)
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[1].getAtomIndices(),
        intraNonBondedContainer2.getAtomIndices()
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[2].getMolecule(),
        &simulationBox.getMolecule(2)
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[2].getAtomIndices(),
        intraNonBondedContainer3.getAtomIndices()
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[3].getMolecule(),
        &simulationBox.getMolecule(3)
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[3].getAtomIndices(),
        intraNonBondedContainer2.getAtomIndices()
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[4].getMolecule(),
        &simulationBox.getMolecule(4)
    );
    EXPECT_EQ(
        intraNonBonded.getIntraNonBondedMaps()[4].getAtomIndices(),
        intraNonBondedContainer3.getAtomIndices()
    );
}

/**
 * @brief test calculate method
 *
 * @details only wrapper for calculate method of IntraNonBondedMap class
 */
TEST_F(TestIntraNonBonded, calculate)
{
    auto molecule = molsys::Molecule{MolType{0}};

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();

    atom1->setPosition({0.0, 0.0, 0.0});
    atom2->setPosition({0.0, 0.0, 11.0});
    atom1->setForce({0.0, 0.0, 0.0});
    atom2->setForce({0.0, 0.0, 0.0});
    atom1->setInternalGlobalVDWType(VdwType{0});
    atom2->setInternalGlobalVDWType(VdwType{1});
    atom1->setAtomType(AtomType{0});
    atom2->setAtomType(AtomType{1});
    atom1->setPartialCharge(0.5);
    atom2->setPartialCharge(-0.5);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.75);

    auto intraNonBondedType =
        intraNonBonded::IntraNonBondedContainer(MolType{0}, {{-1}});
    auto intraNonBondedMap =
        intraNonBonded::IntraNonBondedMap(&molecule, &intraNonBondedType);

    auto coulombPotential = pot::CoulombShiftedPotential(10.0);
    _setNonCoulombPairsMatrix(
        linalg::Matrix<std::shared_ptr<pot::NonCoulombPair>>(2, 2)
    );

    auto nonCoulombPair = pot::LennardJonesPair(
        ExtVdwType(0),
        ExtVdwType(1),
        10.0,
        LJParams{.c6 = 2.0, .c12 = 3.0}
    );
    _setNonCoulombPairsMatrix(0, 1, nonCoulombPair);
    _setNonCoulombPairsMatrix(1, 0, nonCoulombPair);

    auto simulationBox = molsys::SimulationBox();
    simulationBox.setBoxDimensions({10.0, 10.0, 10.0});

    auto physicalData = physicalData::PhysicalData();

    auto intraNonBonded = intraNonBonded::IntraNonBonded();
    intraNonBonded.addIntraNonBondedMap(intraNonBondedMap);

    intraNonBonded.setCoulombPotential(
        std::make_shared<pot::CoulombShiftedPotential>(coulombPotential)
    );
    intraNonBonded.setNonCoulombPotential(
        std::make_shared<pot::ForceFieldNonCoulomb>(*_nonCoulombPotential)
    );
    EXPECT_NO_THROW(intraNonBonded.calculate(simulationBox, physicalData));
}
