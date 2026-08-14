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

#include <cmath>
#include <memory>
#include <vector>

#include "atom.hpp"
#include "constraintSettings.hpp"
#include "exceptions.hpp"
#include "gtest/gtest.h"
#include "mShake.hpp"
#include "mShakeReference.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"

using namespace constraints;
using namespace linearAlgebra;
using namespace simulationBox;

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
 * stretches one bond in the simulationBox::SimulationBox copy, and runs
 * applyMShake. With the fixed UT loop bound, applyMShake must converge without
 * throwing.
 */
TEST(TestMShake, applyMShakeThreeAtomMolecule)
{
    // --- reference shape: equilateral triangle in the xy plane ---
    auto moltype = MoleculeType();
    moltype.setMoltype(1);
    moltype.setName("triangle");
    moltype.setNumberOfAtoms(3);

    auto refAtoms = std::vector<Atom>(3);
    // Atom::initMass (called from MShake::initMShakeReferences) looks the
    // mass up from a name table, so each reference atom needs a valid
    // element name.
    refAtoms[0].setName("H");
    refAtoms[1].setName("H");
    refAtoms[2].setName("H");
    refAtoms[0].setPosition({0.0, 0.0, 0.0});
    refAtoms[1].setPosition({1.0, 0.0, 0.0});
    refAtoms[2].setPosition({0.5, std::sqrt(3.0) / 2.0, 0.0});

    auto mShakeRef = MShakeReference();
    mShakeRef.setMoleculeType(moltype);
    mShakeRef.setAtoms(refAtoms);

    auto mShake = MShake();
    mShake.addMShakeReference(mShakeRef);
    mShake.initMShake();   // builds the (3, 3) mShake inverse matrix

    // --- simulationBox::SimulationBox with one slightly-stretched triangle ---
    auto simBox = SimulationBox();
    simBox.setBoxDimensions({100.0, 100.0, 100.0});

    auto molecule = Molecule();
    molecule.setMoltype(1);
    molecule.setNumberOfAtoms(3);

    const auto refPos0 = Vec3D(0.0, 0.0, 0.0);
    const auto refPos1 = Vec3D(1.0, 0.0, 0.0);
    const auto refPos2 = Vec3D(0.5, std::sqrt(3.0) / 2.0, 0.0);

    auto a1 = std::make_shared<Atom>();
    auto a2 = std::make_shared<Atom>();
    auto a3 = std::make_shared<Atom>();

    a1->setMass(1.0);
    a2->setMass(1.0);
    a3->setMass(1.0);

    // Stretch bond 0-1 by a small amount; M-SHAKE must pull atom 1
    // back along the bond to restore the rigid triangle. The
    // perturbation is intentionally small so the algorithm converges
    // well within the iteration bound.
    a1->setPosition(refPos0);
    a2->setPosition({1.0001, 0.0, 0.0});
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
    settings::ConstraintSettings::setMShakeMaxIter(100);

    EXPECT_NO_THROW(mShake.applyMShake(simBox));

    // After convergence the perturbed bond 0-1 must be back to the
    // reference length 1.0 within the requested tolerance.
    const auto pos0   = molecule.getAtomPosition(0);
    const auto pos1   = molecule.getAtomPosition(1);
    const auto bond01 = norm(pos1 - pos0);
    EXPECT_NEAR(bond01, 1.0, 1.0e-5);
}

/**
 * @brief if the solver cannot reach tolerance within `shakeIterations` steps,
 * the routine must throw MShakeException rather than looping forever or
 * silently giving up.
 */
TEST(TestMShake, applyMShakeThrowsWhenIterationLimitTooSmall)
{
    auto moltype = MoleculeType();
    moltype.setMoltype(1);
    moltype.setName("triangle");
    moltype.setNumberOfAtoms(3);

    auto refAtoms = std::vector<Atom>(3);
    refAtoms[0].setName("H");
    refAtoms[1].setName("H");
    refAtoms[2].setName("H");
    refAtoms[0].setPosition({0.0, 0.0, 0.0});
    refAtoms[1].setPosition({1.0, 0.0, 0.0});
    refAtoms[2].setPosition({0.5, std::sqrt(3.0) / 2.0, 0.0});

    auto mShakeRef = MShakeReference();
    mShakeRef.setMoleculeType(moltype);
    mShakeRef.setAtoms(refAtoms);

    auto mShake = MShake();
    mShake.addMShakeReference(mShakeRef);
    mShake.initMShake();

    auto simBox = SimulationBox();
    simBox.setBoxDimensions({100.0, 100.0, 100.0});

    auto molecule = Molecule();
    molecule.setMoltype(1);
    molecule.setNumberOfAtoms(3);

    const auto refPos0 = Vec3D(0.0, 0.0, 0.0);
    const auto refPos1 = Vec3D(1.0, 0.0, 0.0);
    const auto refPos2 = Vec3D(0.5, std::sqrt(3.0) / 2.0, 0.0);

    auto a1 = std::make_shared<Atom>();
    auto a2 = std::make_shared<Atom>();
    auto a3 = std::make_shared<Atom>();
    a1->setMass(1.0);
    a2->setMass(1.0);
    a3->setMass(1.0);
    // Large perturbation so the solver cannot converge in 1 iteration.
    a1->setPosition(refPos0);
    a2->setPosition({1.5, 0.0, 0.0});
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
    settings::ConstraintSettings::setMShakeMaxIter(1);
    settings::ConstraintSettings::setMShakeTolerance(-1.0);

    EXPECT_THROW(mShake.applyMShake(simBox), customException::MShakeException);
}
