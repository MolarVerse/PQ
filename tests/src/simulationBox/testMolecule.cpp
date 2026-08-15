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

#include "testMolecule.hpp"

#include "gtest/gtest.h"          // for Message, TestPartResult
#include "manostatSettings.hpp"   // for ManostatSettings
#include "mathUtilities.hpp"      // for compare
#include "moleculeType.hpp"       // for MoleculeType
#include "orthorhombicBox.hpp"    // for OrthorhombicBox

TEST_F(TestMolecule, calculateCenterOfMass)
{
    const linearAlgebra::Vec3D boxDimensions = {10.0, 10.0, 10.0};
    const linearAlgebra::Vec3D centerOfMass  = {1.0 / 3.0, 1.0 / 2.0, 0.0};
    molsys::OrthorhombicBox    box;
    box.setBoxDimensions(boxDimensions);

    _molecule->calculateCenterOfMass(box);
    EXPECT_EQ(_molecule->getCenterOfMass(), centerOfMass);
}

TEST_F(TestMolecule, scaleAtoms)
{
    const linearAlgebra::tensor3D scale =
        diagonalMatrix(linearAlgebra::Vec3D{1.0, 2.0, 3.0});
    const linearAlgebra::Vec3D atomPosition1 = _molecule->getAtomPosition(0);
    const linearAlgebra::Vec3D atomPosition2 = _molecule->getAtomPosition(1);
    const linearAlgebra::Vec3D atomPosition3 = _molecule->getAtomPosition(2);

    molsys::OrthorhombicBox box;
    box.setBoxDimensions({10.0, 10.0, 10.0});

    _molecule->calculateCenterOfMass(box);

    const auto centerOfMassBeforeScaling = _molecule->getCenterOfMass();
    const linearAlgebra::Vec3D shift =
        centerOfMassBeforeScaling * (diagonal(scale) - 1.0);

    _molecule->scale(scale, box);

    EXPECT_EQ(_molecule->getAtomPosition(0), atomPosition1 + shift);
    EXPECT_EQ(_molecule->getAtomPosition(1), atomPosition2 + shift);
    EXPECT_EQ(_molecule->getAtomPosition(2), atomPosition3 + shift);
}

TEST_F(TestMolecule, scaleAtomsWrapsIntoBox)
{
    const linearAlgebra::tensor3D scale =
        diagonalMatrix(linearAlgebra::Vec3D{0.5, 0.5, 0.5});

    molsys::OrthorhombicBox box;
    box.setBoxDimensions({2.0, 2.0, 2.0});

    _molecule->setAtomPosition(0, {0.9, 0.0, 0.0});
    _molecule->setAtomPosition(1, {-0.9, 0.0, 0.0});
    _molecule->setAtomPosition(2, {0.9, 0.1, 0.0});
    _molecule->calculateCenterOfMass(box);

    const auto centerOfMassBeforeScaling = _molecule->getCenterOfMass();
    const auto shift = centerOfMassBeforeScaling * (diagonal(scale) - 1.0);

    box.scaleBox(scale);
    _molecule->scale(scale, box);

    auto expectedPosition0 = linearAlgebra::Vec3D{0.9, 0.0, 0.0} + shift;
    auto expectedPosition1 = linearAlgebra::Vec3D{-0.9, 0.0, 0.0} + shift;
    auto expectedPosition2 = linearAlgebra::Vec3D{0.9, 0.1, 0.0} + shift;
    box.applyPBC(expectedPosition0);
    box.applyPBC(expectedPosition1);
    box.applyPBC(expectedPosition2);

    EXPECT_EQ(_molecule->getAtomPosition(0), expectedPosition0);
    EXPECT_EQ(_molecule->getAtomPosition(1), expectedPosition1);
    EXPECT_EQ(_molecule->getAtomPosition(2), expectedPosition2);
}

TEST_F(TestMolecule, scaleVelocityPreservesInternalVelocities)
{
    settings::ManostatSettings::setIsotropy(settings::Isotropy::ISOTROPIC);

    const linearAlgebra::tensor3D scale =
        diagonalMatrix(linearAlgebra::Vec3D{0.5, 0.25, 2.0});

    molsys::OrthorhombicBox box;
    box.setBoxDimensions({10.0, 10.0, 10.0});

    const auto relativeVelocity10 =
        _molecule->getAtomVelocity(1) - _molecule->getAtomVelocity(0);
    const auto relativeVelocity20 =
        _molecule->getAtomVelocity(2) - _molecule->getAtomVelocity(0);

    const auto centerOfMassVelocity = (1.0 * _molecule->getAtomVelocity(0) +
                                       2.0 * _molecule->getAtomVelocity(1) +
                                       3.0 * _molecule->getAtomVelocity(2)) /
                                      6.0;

    _molecule->scaleVelocity(scale, box);

    const auto scaledCenterOfMassVelocity =
        (1.0 * _molecule->getAtomVelocity(0) +
         2.0 * _molecule->getAtomVelocity(1) +
         3.0 * _molecule->getAtomVelocity(2)) /
        6.0;

    EXPECT_TRUE(
        utilities::compare(
            scaledCenterOfMassVelocity,
            scale * centerOfMassVelocity,
            1e-12
        )
    );
    EXPECT_TRUE(
        utilities::compare(
            _molecule->getAtomVelocity(1) - _molecule->getAtomVelocity(0),
            relativeVelocity10,
            1e-12
        )
    );
    EXPECT_TRUE(
        utilities::compare(
            _molecule->getAtomVelocity(2) - _molecule->getAtomVelocity(0),
            relativeVelocity20,
            1e-12
        )
    );
}

TEST_F(TestMolecule, setAtomForceToZero)
{
    _molecule->setAtomForcesToZero();
    EXPECT_EQ(_molecule->getAtomForce(0), linearAlgebra::Vec3D());
    EXPECT_EQ(_molecule->getAtomForce(1), linearAlgebra::Vec3D());
    EXPECT_EQ(_molecule->getAtomForce(2), linearAlgebra::Vec3D());
}

TEST_F(TestMolecule, getNumberOfAtomTypes)
{
    EXPECT_EQ(_molecule->getNumberOfAtomTypes(), 2);
}

TEST_F(TestMolecule, getNumberOfAtomTypesCountsNonAdjacentDuplicates)
{
    auto molecule = molsys::Molecule();
    molecule.setNumberOfAtoms(3);

    const auto atom1 = std::make_shared<molsys::Atom>();
    const auto atom2 = std::make_shared<molsys::Atom>();
    const auto atom3 = std::make_shared<molsys::Atom>();

    atom1->setExternalAtomType(1);
    atom2->setExternalAtomType(2);
    atom3->setExternalAtomType(1);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    EXPECT_EQ(molecule.getNumberOfAtomTypes(), 2);
}

TEST_F(TestMolecule, moleculeTypeCountsNonAdjacentDuplicates)
{
    auto moleculeType = molsys::MoleculeType();

    moleculeType.addAtomType(1);
    moleculeType.addAtomType(2);
    moleculeType.addAtomType(1);

    EXPECT_EQ(moleculeType.getNumberOfAtomTypes(), 2);
}
