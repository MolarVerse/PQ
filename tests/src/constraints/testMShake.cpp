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
#include <string>
#include <vector>

#include "atom.hpp"
#include "constraintSettings.hpp"
#include "exceptions.hpp"
#include "mShake.hpp"
#include "mShakeReference.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "simulationBox.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"

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
 * stretches one bond in the molsys::SimulationBox copy, and runs
 * applyMShake. With the fixed UT loop bound, applyMShake must converge without
 * throwing.
 */
TEST(TestMShake, applyMShakeThreeAtomMolecule)
{
    // --- reference shape: equilateral triangle in the xy plane ---
    auto moltype = molsys::MoleculeType();
    moltype.setMoltype(MolType{1});
    moltype.setName("triangle");
    moltype.setNumberOfAtoms(3);

    auto refAtoms = std::vector<molsys::Atom>(3);
    // Atom::initMass (called from constraints::MShake::initMShakeReferences)
    // looks the mass up from a name table, so each reference atom needs a valid
    // element name.
    refAtoms[0].setName("H");
    refAtoms[1].setName("H");
    refAtoms[2].setName("H");
    refAtoms[0].setPosition({0.0, 0.0, 0.0});
    refAtoms[1].setPosition({1.0, 0.0, 0.0});
    refAtoms[2].setPosition({0.5, std::numbers::sqrt3 / 2.0, 0.0});

    auto mShakeRef = constraints::MShakeReference();
    mShakeRef.setMoleculeType(moltype);
    mShakeRef.setAtoms(refAtoms);

    auto mShake = constraints::MShake();
    mShake.addMShakeReference(mShakeRef);
    mShake.initMShake();   // builds the (3, 3) mShake inverse matrix

    // --- molsys::SimulationBox with one slightly-stretched triangle ---
    auto simBox = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{{100.0, 100.0, 100.0}};
    simBox.setBox(orthoBox, false);

    auto molecule = molsys::Molecule();
    molecule.setMoltype(MolType{1});

    const auto refPos0 = linalg::Vec3D(0.0, 0.0, 0.0);
    const auto refPos1 = linalg::Vec3D(1.0, 0.0, 0.0);
    const auto refPos2 = linalg::Vec3D(0.5, std::numbers::sqrt3 / 2.0, 0.0);

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();
    auto atom3 = std::make_shared<molsys::Atom>();

    atom1->setMass(1.0);
    atom2->setMass(1.0);
    atom3->setMass(1.0);

    // Stretch bond 0-1 by a small amount; M-SHAKE must pull atom 1
    // back along the bond to restore the rigid triangle. The
    // perturbation is intentionally small so the algorithm converges
    // well within the iteration bound.
    atom1->setPosition(refPos0);
    atom2->setPosition({1.0001, 0.0, 0.0});
    atom3->setPosition(refPos2);

    atom1->setPositionOld(refPos0);
    atom2->setPositionOld(refPos1);
    atom3->setPositionOld(refPos2);

    atom1->setVelocity({0.0, 0.0, 0.0});
    atom2->setVelocity({0.0, 0.0, 0.0});
    atom3->setVelocity({0.0, 0.0, 0.0});

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    simBox.addMolecule(molecule);

    settings::TimingsSettings::setTimeStep(0.5);
    settings::ConstraintSettings::setMShakeMaxIter(100);

    EXPECT_NO_THROW(mShake.applyMShake(simBox));

    // After convergence the perturbed bond 0-1 must be back to the
    // reference length 1.0 within the requested tolerance.
    const auto pos0   = molecule.getAtomPosition(AtomIndex{0});
    const auto pos1   = molecule.getAtomPosition(AtomIndex{1});
    const auto bond01 = norm(pos1 - pos0);
    EXPECT_NEAR(bond01, 1.0, 1.0e-5);
}

/**
 * @brief M-RATTLE must remove the relative velocity along every bond of a
 * rigid water molecule (r_ij . (v_i - v_j) = 0), without changing the
 * total momentum.
 */
TEST(TestMShake, applyMRattleRemovesBondVelocities)
{
    auto moltype = molsys::MoleculeType();
    moltype.setMoltype(MolType{1});
    moltype.setName("water");
    moltype.setNumberOfAtoms(3);

    const auto names  = std::vector<std::string>{"O", "H", "H"};
    const auto refPos = std::vector<linalg::Vec3D>{
        {0.0, 0.0, 0.0},
        {0.585882, 0.756950, 0.0},
        {0.585882, -0.756950, 0.0}
    };

    auto refAtoms = std::vector<molsys::Atom>(3);
    for (size_t i = 0; i < 3; ++i)
    {
        refAtoms[i].setName(names[i]);
        refAtoms[i].setPosition(refPos[i]);
    }

    auto mShakeRef = constraints::MShakeReference();
    mShakeRef.setMoleculeType(moltype);
    mShakeRef.setAtoms(refAtoms);

    auto mShake = constraints::MShake();
    mShake.addMShakeReference(mShakeRef);
    mShake.initMShake();

    auto simBox = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{{100.0, 100.0, 100.0}};
    simBox.setBox(orthoBox, false);

    auto molecule = molsys::Molecule();
    molecule.setMoltype(MolType{1});

    // same geometry as the reference, rotated and shifted
    const auto pos = std::vector<linalg::Vec3D>{
        {10.0, 10.0, 10.0},
        {10.0, 10.585882, 10.756950},
        {10.0, 10.585882, 9.243050}
    };
    const auto vel = std::vector<linalg::Vec3D>{
        {0.3, -0.2, 0.1},
        {-1.1, 2.0, 0.7},
        {0.9, -1.4, -2.3}
    };

    auto atoms = std::vector<std::shared_ptr<molsys::Atom>>();
    for (size_t i = 0; i < 3; ++i)
    {
        auto atom = std::make_shared<molsys::Atom>();
        atom->setName(names[i]);
        atom->initMass();
        atom->setPosition(pos[i]);
        atom->setVelocity(vel[i]);
        molecule.addAtom(atom);
        atoms.push_back(atom);
    }

    simBox.addMolecule(molecule);

    auto momentumBefore = linalg::Vec3D(0.0, 0.0, 0.0);
    for (const auto &atom : atoms)
        momentumBefore += atom->getMass() * atom->getVelocity();

    mShake.applyMRattle(simBox);

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = i + 1; j < 3; ++j)
        {
            const auto bond = atoms[i]->getPosition() - atoms[j]->getPosition();
            const auto dVel = atoms[i]->getVelocity() - atoms[j]->getVelocity();
            EXPECT_NEAR(dot(bond, dVel), 0.0, 1.0e-10);
        }

    auto momentumAfter = linalg::Vec3D(0.0, 0.0, 0.0);
    for (const auto &atom : atoms)
        momentumAfter += atom->getMass() * atom->getVelocity();

    for (size_t k = 0; k < 3; ++k)
        EXPECT_NEAR(momentumAfter[k], momentumBefore[k], 1.0e-10);
}

/**
 * @brief if the solver cannot reach tolerance within `shakeIterations` steps,
 * the routine must throw MShakeException rather than looping forever or
 * silently giving up.
 */
TEST(TestMShake, applyMShakeThrowsWhenIterationLimitTooSmall)
{
    auto moltype = molsys::MoleculeType();
    moltype.setMoltype(MolType{1});
    moltype.setName("triangle");
    moltype.setNumberOfAtoms(3);

    auto refAtoms = std::vector<molsys::Atom>(3);
    refAtoms[0].setName("H");
    refAtoms[1].setName("H");
    refAtoms[2].setName("H");
    refAtoms[0].setPosition({0.0, 0.0, 0.0});
    refAtoms[1].setPosition({1.0, 0.0, 0.0});
    refAtoms[2].setPosition({0.5, std::numbers::sqrt3 / 2.0, 0.0});

    auto mShakeRef = constraints::MShakeReference();
    mShakeRef.setMoleculeType(moltype);
    mShakeRef.setAtoms(refAtoms);

    auto mShake = constraints::MShake();
    mShake.addMShakeReference(mShakeRef);
    mShake.initMShake();

    auto simBox = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{{100.0, 100.0, 100.0}};
    simBox.setBox(orthoBox, false);

    auto molecule = molsys::Molecule();
    molecule.setMoltype(MolType{1});

    const auto refPos0 = linalg::Vec3D(0.0, 0.0, 0.0);
    const auto refPos1 = linalg::Vec3D(1.0, 0.0, 0.0);
    const auto refPos2 = linalg::Vec3D(0.5, std::numbers::sqrt3 / 2.0, 0.0);

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();
    auto atom3 = std::make_shared<molsys::Atom>();
    atom1->setMass(1.0);
    atom2->setMass(1.0);
    atom3->setMass(1.0);
    // Large perturbation so the solver cannot converge in 1 iteration.
    atom1->setPosition(refPos0);
    atom2->setPosition({1.5, 0.0, 0.0});
    atom3->setPosition(refPos2);
    atom1->setPositionOld(refPos0);
    atom2->setPositionOld(refPos1);
    atom3->setPositionOld(refPos2);
    atom1->setVelocity({0.0, 0.0, 0.0});
    atom2->setVelocity({0.0, 0.0, 0.0});
    atom3->setVelocity({0.0, 0.0, 0.0});
    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);
    simBox.addMolecule(molecule);

    settings::TimingsSettings::setTimeStep(0.5);
    settings::ConstraintSettings::setMShakeMaxIter(1);
    settings::ConstraintSettings::setMShakeTolerance(-1.0);

    EXPECT_THROW_MSG(
        mShake.applyMShake(simBox),
        exc::MShakeException,
        "M-Shake did not converge within 1 iterations for molecule type "
        "MolType(1)"
    );
}
