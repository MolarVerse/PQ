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

#include <cmath>
#include <memory>
#include <vector>

#include "atom.hpp"
#include "gtest/gtest.h"
#include "mShake.hpp"
#include "mShakeReference.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"
#include "vector3d.hpp"

/**
 * @brief regression test for the M-SHAKE inner-loop bound (3-atom molecule).
 *
 * The (k, l) bond-pair inner loop in applyMShake used to be rectangular
 * ((nAtoms - 1)^2) instead of upper-triangular (n * (n - 1) / 2). For
 * nAtoms = 2 these coincide (1 == 1), so existing diatomic tests never
 * exercised the OOB read/write into the (nBonds, nBonds) mShakeMatrix
 * and the nBonds-sized bondsUnconstrained vector. nAtoms = 3 is the
 * smallest case that hits the bug: 4 inner iterations vs. 3 bonds, so
 * one element past the matrix and vector is touched per outer iteration.
 *
 * This test sets up a rigid equilateral triangular reference molecule,
 * stretches one bond in the SimBox copy, and runs applyMShake. With the
 * fixed UT loop bound, applyMShake must converge without throwing.
 */
TEST(TestMShake, applyMShake_threeAtomMolecule)
{
    // --- reference shape: equilateral triangle in the xy plane ---
    auto moltype = simulationBox::MoleculeType();
    moltype.setMoltype(1);
    moltype.setName("triangle");
    moltype.setNumberOfAtoms(3);

    auto refAtoms = std::vector<simulationBox::Atom>(3);
    refAtoms[0].setPosition({0.0, 0.0, 0.0});
    refAtoms[1].setPosition({1.0, 0.0, 0.0});
    refAtoms[2].setPosition({0.5, std::sqrt(3.0) / 2.0, 0.0});
    for (auto &atom : refAtoms) atom.setMass(1.0);

    auto mShakeRef = constraints::MShakeReference();
    mShakeRef.setMoleculeType(moltype);
    mShakeRef.setAtoms(refAtoms);

    auto mShake = constraints::MShake();
    mShake.addMShakeReference(mShakeRef);
    mShake.initMShake();   // builds the (3, 3) mShake inverse matrix

    // --- SimBox with one slightly-stretched triangle ---
    auto simBox = simulationBox::SimulationBox();
    simBox.setBoxDimensions({100.0, 100.0, 100.0});

    auto molecule = simulationBox::Molecule();
    molecule.setMoltype(1);
    molecule.setNumberOfAtoms(3);

    const auto refPos0 = linearAlgebra::Vec3D(0.0, 0.0, 0.0);
    const auto refPos1 = linearAlgebra::Vec3D(1.0, 0.0, 0.0);
    const auto refPos2 =
        linearAlgebra::Vec3D(0.5, std::sqrt(3.0) / 2.0, 0.0);

    auto a1 = std::make_shared<simulationBox::Atom>();
    auto a2 = std::make_shared<simulationBox::Atom>();
    auto a3 = std::make_shared<simulationBox::Atom>();

    a1->setMass(1.0);
    a2->setMass(1.0);
    a3->setMass(1.0);

    // Stretch bond 1-2 slightly; M-SHAKE must pull atom 2 back.
    a1->setPosition(refPos0);
    a2->setPosition({1.05, 0.0, 0.0});
    a3->setPosition(refPos2);

    a1->setPositionOld(refPos0);
    a2->setPositionOld(refPos1);
    a3->setPositionOld(refPos2);

    a1->setVelocity({0.0, 0.0, 0.0});
    a2->setVelocity({0.0, 0.0, 0.0});
    a3->setVelocity({0.0, 0.0, 0.0});

    molecule.addAtom(a1);
    molecule.addAtom(a2);
    molecule.addAtom(a3);

    simBox.addMolecule(molecule);

    settings::TimingsSettings::setTimeStep(0.5);

    // With the upper-triangular fix this converges; with the previous
    // rectangular loop, every iteration writes past the (3, 3) matrix
    // and reads past the 3-element bondsUnconstrained vector.
    EXPECT_NO_THROW(mShake.applyMShake(1.0e-6, simBox));
}
