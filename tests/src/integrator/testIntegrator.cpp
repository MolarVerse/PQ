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

#include "testIntegrator.hpp"

#include <vector>

#include "units.hpp"

namespace
{
    // A/s gained from 0.5 * F * dt / m for F = 1 kcal/mol/A, dt = 1 fs,
    // m = 1 g/mol
    const double velocityVerletFactor =
        (0.5 * (1.0 * kcal_per_mol / angstrom) * (1.0 * fs) / (1.0 * g_per_mol))
            .numerical_value_in(angstrom / s);
}   // namespace

/**
 * @brief tests function integrate velocities of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, integrateVelocities)
{
    integrator::Integrator::integrateVelocities(_box->getAtoms()[0].get());
    EXPECT_EQ(
        _molecule1->getAtomVelocity(AtomIndex{0}),
        linalg::Vec3D(0.0, 0.0, 0.0) * angstrom / s
    );
    integrator::Integrator::integrateVelocities(_box->getAtoms()[1].get());

    const auto vel = _molecule1->getAtomVelocity(AtomIndex{1});
    EXPECT_DOUBLE_EQ(
        vel[0].numerical_value_in(angstrom / s),
        1.0 + (0.1 * 0.5 * velocityVerletFactor)
    );
    EXPECT_DOUBLE_EQ(
        vel[1].numerical_value_in(angstrom / s),
        2.0 + (0.1 * 1.5 * velocityVerletFactor)
    );
    EXPECT_DOUBLE_EQ(
        vel[2].numerical_value_in(angstrom / s),
        3.0 + (0.1 * 2.5 * velocityVerletFactor)
    );
}

/**
 * @brief tests function integrate positions of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, integratePositions)
{
    integrator::Integrator::integratePositions(
        _box->getAtoms()[0].get(),
        *_box
    );
    EXPECT_EQ(
        _molecule1->getAtomPosition(AtomIndex{0}),
        linalg::Vec3D(0.0, 0.0, 0.0) * angstrom
    );
    integrator::Integrator::integratePositions(
        _box->getAtoms()[1].get(),
        *_box
    );
    const auto pos = _molecule1->getAtomPosition(AtomIndex{1});
    EXPECT_DOUBLE_EQ(
        pos[0].numerical_value_in(angstrom),
        1.0 + (0.1 * (1.0 * fs).numerical_value_in(s))
    );
    EXPECT_DOUBLE_EQ(
        pos[1].numerical_value_in(angstrom),
        1.0 + (0.1 * (2.0 * fs).numerical_value_in(s))
    );
    EXPECT_DOUBLE_EQ(
        pos[2].numerical_value_in(angstrom),
        1.0 + (0.1 * (3.0 * fs).numerical_value_in(s))
    );
}

/**
 * @brief tests function firstStep of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, firstStep)
{
    _integrator->firstStep(*_box);

    const auto molecule = _box->getMolecules()[0];
    EXPECT_EQ(
        molecule.getAtomVelocity(AtomIndex{0}),
        linalg::Vec3D(0.0, 0.0, 0.0) * angstrom / s
    );

    auto velocities = linalg::Vec3D(1.0, 2.0, 3.0);
    velocities += 0.1 * linalg::Vec3D(0.5, 1.5, 2.5) * velocityVerletFactor;

    const auto vel = molecule.getAtomVelocity(AtomIndex{1});
    EXPECT_DOUBLE_EQ(vel[0].numerical_value_in(angstrom / s), velocities[0]);
    EXPECT_DOUBLE_EQ(vel[1].numerical_value_in(angstrom / s), velocities[1]);
    EXPECT_DOUBLE_EQ(vel[2].numerical_value_in(angstrom / s), velocities[2]);

    EXPECT_DOUBLE_EQ(
        molecule.getAtomPosition(AtomIndex{0})[0].numerical_value_in(angstrom),
        0.0
    );

    const auto pos = molecule.getAtomPosition(AtomIndex{1});
    EXPECT_DOUBLE_EQ(
        pos[0].numerical_value_in(angstrom),
        1.0 + (0.1 * velocities[0] * (1.0 * fs).numerical_value_in(s))
    );
    EXPECT_DOUBLE_EQ(
        pos[1].numerical_value_in(angstrom),
        1.0 + (0.1 * velocities[1] * (1.0 * fs).numerical_value_in(s))
    );
    EXPECT_DOUBLE_EQ(
        pos[2].numerical_value_in(angstrom),
        1.0 + (0.1 * velocities[2] * (1.0 * fs).numerical_value_in(s))
    );

    EXPECT_EQ(molecule.getAtomForce(AtomIndex{0}), Force3D::zero());
    EXPECT_EQ(molecule.getAtomForce(AtomIndex{1}), Force3D::zero());

    EXPECT_TRUE(
        molecule.getCenterOfMass() != linalg::Vec3D(0.0, 0.0, 0.0) * angstrom
    );
}

/**
 * @brief tests function secondStep of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, secondStep)
{
    _integrator->secondStep(*_box);

    const auto molecule = _box->getMolecules()[0];
    EXPECT_EQ(
        molecule.getAtomVelocity(AtomIndex{0}),
        linalg::Vec3D(0.0, 0.0, 0.0) * angstrom / s
    );

    const auto vel = molecule.getAtomVelocity(AtomIndex{1});
    EXPECT_DOUBLE_EQ(
        vel[0].numerical_value_in(angstrom / s),
        1.0 + (0.1 * 0.5 * velocityVerletFactor)
    );
    EXPECT_DOUBLE_EQ(
        vel[1].numerical_value_in(angstrom / s),
        2.0 + (0.1 * 1.5 * velocityVerletFactor)
    );
    EXPECT_DOUBLE_EQ(
        vel[2].numerical_value_in(angstrom / s),
        3.0 + (0.1 * 2.5 * velocityVerletFactor)
    );
}

TEST_F(TestIntegrator, velocityVerletZeroForceKeepsFreeParticleVelocity)
{
    settings::TimingsSettings::setTimeStep(0.2 * fs);

    auto *atom = _box->getAtoms()[1].get();
    atom->setPosition(linalg::Vec3D(1.0, 2.0, 3.0) * angstrom);
    atom->setVelocity(linalg::Vec3D(2.0, -3.0, 4.0) * angstrom / s);
    atom->setForce(Force3D::zero());

    const auto initialPosition =
        atom->getPosition().numerical_value_in(angstrom);
    const auto initialVelocity = atom->getVelocity();

    _integrator->firstStep(*_box);
    _integrator->secondStep(*_box);

    const auto expectedPosition =
        initialPosition + 0.2 *
                              initialVelocity.numerical_value_in(angstrom / s) *
                              (1.0 * fs).numerical_value_in(s);

    EXPECT_DOUBLE_EQ(
        atom->getVelocity()[0].numerical_value_in(angstrom / s),
        initialVelocity[0].numerical_value_in(angstrom / s)
    );
    EXPECT_DOUBLE_EQ(
        atom->getVelocity()[1].numerical_value_in(angstrom / s),
        initialVelocity[1].numerical_value_in(angstrom / s)
    );
    EXPECT_DOUBLE_EQ(
        atom->getVelocity()[2].numerical_value_in(angstrom / s),
        initialVelocity[2].numerical_value_in(angstrom / s)
    );

    EXPECT_DOUBLE_EQ(
        atom->getPosition()[0].numerical_value_in(angstrom),
        expectedPosition[0]
    );
    EXPECT_DOUBLE_EQ(
        atom->getPosition()[1].numerical_value_in(angstrom),
        expectedPosition[1]
    );
    EXPECT_DOUBLE_EQ(
        atom->getPosition()[2].numerical_value_in(angstrom),
        expectedPosition[2]
    );
}
