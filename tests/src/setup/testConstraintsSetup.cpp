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

#include "atom.hpp"
#include "bondConstraint.hpp"
#include "constraintSettings.hpp"
#include "constraints.hpp"
#include "constraintsSetup.hpp"
#include "engine.hpp"
#include "fileSettings.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "simulationBox.hpp"
#include "testSetup.hpp"

namespace
{
    constexpr MolType kMShakeMolType{1};

    /**
     * @brief adds a two-atom molecule to the engine's simulation box and
     * sets up a 10x10x10 orthorhombic box so that distance/PBC
     * calculations are well defined
     */
    void addDiatomicMolecule(
        engine::Engine      &engine,
        const linalg::Vec3D &position1,
        const linalg::Vec3D &position2
    )
    {
        auto &simBox = engine.getSimulationBox();

        molsys::Molecule molecule;

        auto atom1 = std::make_shared<molsys::Atom>();
        atom1->setPosition(position1);
        atom1->setMass(1.0);
        molecule.addAtom(atom1);
        simBox.addAtom(atom1);

        auto atom2 = std::make_shared<molsys::Atom>();
        atom2->setPosition(position2);
        atom2->setMass(1.0);
        molecule.addAtom(atom2);
        simBox.addAtom(atom2);

        simBox.addMolecule(molecule);

        molsys::OrthorhombicBox orthoBox{{10.0, 10.0, 10.0}};
        simBox.setBox(orthoBox, false);
    }

    /**
     * @brief adds a bond constraint between the two atoms of the last
     * molecule added to the engine's simulation box
     */
    void addBondConstraintToLastMolecule(
        engine::Engine &engine,
        double          bondLength
    )
    {
        auto &simBox   = engine.getSimulationBox();
        auto &molecule = simBox.getMolecules().back();

        const constraints::BondConstraint bondConstraint(
            &molecule,
            &molecule,
            AtomIndex{0},
            AtomIndex{1},
            bondLength
        );

        engine.getConstraints()->addBondConstraint(bondConstraint);
    }

    /**
     * @brief registers the M-Shake molecule type (matching
     * tests/data/constraintsSetup/mshake.dat) with the engine and adds a
     * single real molecule of that type to the simulation box
     */
    void setupMShakeMoleculeType(engine::Engine &engine)
    {
        auto &simBox = engine.getSimulationBox();

        molsys::MoleculeType moleculeType{kMShakeMolType};
        moleculeType.setNumberOfAtoms(2);
        moleculeType.addAtomName("H");
        moleculeType.addAtomName("O");
        simBox.addMoleculeType(moleculeType);

        molsys::Molecule molecule;
        molecule.setMoltype(kMShakeMolType);

        auto atom1 = std::make_shared<molsys::Atom>();
        atom1->setPosition({0.0, 0.0, 0.0});
        molecule.addAtom(atom1);
        simBox.addAtom(atom1);

        auto atom2 = std::make_shared<molsys::Atom>();
        atom2->setPosition({1.0, 0.0, 0.0});
        molecule.addAtom(atom2);
        simBox.addAtom(atom2);

        simBox.addMolecule(molecule);

        molsys::OrthorhombicBox orthoBox{{10.0, 10.0, 10.0}};
        simBox.setBox(orthoBox, false);

        settings::FileSettings::setMShakeFileName(
            "data/constraintsSetup/mshake.dat"
        );
    }
}   // namespace

/**
 * @brief tests that setup() calculates the shake reference bond vector and
 * reduces the degrees of freedom by the number of shake bond constraints
 */
TEST_F(TestSetup, constraintsSetupComputesRefBondLengthsForShake)
{
    _settings->constraintSettings.activateShake();

    addDiatomicMolecule(*_engine, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
    addBondConstraintToLastMolecule(*_engine, 1.0);

    auto &simBox = _engine->getSimulationBox();
    simBox.setDegreesOfFreedom(10);

    setup::ConstraintsSetup constraintsSetup(*_engine);
    EXPECT_NO_THROW(constraintsSetup.setup());

    const auto &constraints = _engine->getConstraints();

    EXPECT_EQ(
        constraints->getBondConstraints()[0].getShakeDistanceRef(),
        linalg::Vec3D(-1.0, 0.0, 0.0)
    );
    EXPECT_EQ(constraints->getNumberOfBondConstraints(), 1);
    EXPECT_EQ(simBox.getDegreesOfFreedom(), 9);
}

/**
 * @brief tests that setup() reads the M-Shake file, initializes M-Shake and
 * reduces the degrees of freedom by the number of M-Shake bond constraints
 */
TEST_F(TestSetup, constraintsSetupRunsMShakeAndReducesDegreesOfFreedom)
{
    _settings->constraintSettings.activateMShake();

    setupMShakeMoleculeType(*_engine);

    auto &simBox = _engine->getSimulationBox();
    simBox.setDegreesOfFreedom(10);

    setup::ConstraintsSetup constraintsSetup(*_engine);
    EXPECT_NO_THROW(constraintsSetup.setup());

    const auto &constraints = _engine->getConstraints();

    EXPECT_EQ(constraints->getMShakeReferences().size(), 1);
    EXPECT_EQ(constraints->getNumberOfMShakeConstraints(simBox), 1);
    EXPECT_EQ(simBox.getDegreesOfFreedom(), 9);
}

/**
 * @brief tests that setup() is a no-op with respect to constraints and
 * degrees of freedom when neither shake nor M-Shake is activated and no
 * bond constraints have been added
 */
TEST_F(TestSetup, constraintsSetupNoOpWhenNoConstraintsActive)
{
    auto &simBox = _engine->getSimulationBox();
    simBox.setDegreesOfFreedom(10);

    setup::ConstraintsSetup constraintsSetup(*_engine);
    EXPECT_NO_THROW(constraintsSetup.setup());

    const auto &constraints = _engine->getConstraints();

    EXPECT_EQ(constraints->getNumberOfBondConstraints(), 0);
    EXPECT_TRUE(constraints->getMShakeReferences().empty());
    EXPECT_EQ(simBox.getDegreesOfFreedom(), 10);
}

/**
 * @brief tests that the setupConstraints wrapper function does not touch the
 * degrees of freedom when no constraint algorithm is activated
 */
TEST_F(TestSetup, setupConstraintsWrapperSkipsWhenInactive)
{
    auto &simBox = _engine->getSimulationBox();
    simBox.setDegreesOfFreedom(5);

    EXPECT_NO_THROW(setup::setupConstraints(*_engine));

    EXPECT_EQ(simBox.getDegreesOfFreedom(), 5);
}

/**
 * @brief tests that the setupConstraints wrapper function delegates to
 * ConstraintsSetup and reduces the degrees of freedom when shake is active
 */
TEST_F(TestSetup, setupConstraintsWrapperRunsWhenShakeActive)
{
    _settings->constraintSettings.activateShake();

    addDiatomicMolecule(*_engine, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
    addBondConstraintToLastMolecule(*_engine, 1.0);

    auto &simBox = _engine->getSimulationBox();
    simBox.setDegreesOfFreedom(10);

    EXPECT_NO_THROW(setup::setupConstraints(*_engine));

    const auto &constraints = _engine->getConstraints();

    EXPECT_EQ(
        constraints->getBondConstraints()[0].getShakeDistanceRef(),
        linalg::Vec3D(-1.0, 0.0, 0.0)
    );
    EXPECT_EQ(simBox.getDegreesOfFreedom(), 9);
}
