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

#include "testBondConstraint.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "exceptions.hpp"
#include "throwWithMessage.hpp"
#include "unit_types.hpp"
#include "units.hpp"

/**
 * @brief tests calculation of bond constraint ref bond length
 *
 */
TEST_F(TestBondConstraint, calcRefBondLength)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    EXPECT_EQ(
        _bondConstraint->getShakeDistanceRef(),
        linalg::Vec3D(0.0, -1.0, -2.0) * angstrom
    );
}

/**
 * @brief tests calculation of bond constraint delta bond length
 *
 */
TEST_F(TestBondConstraint, calculateDistanceDelta)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    const auto distanceSquared =
        normSquared(linalg::Vec3D(0.0, -1.0, -2.0)) * angstrom * angstrom;
    const auto targetBondLengthSquared = _targetBondLength * _targetBondLength;
    EXPECT_EQ(
        _bondConstraint->calculateDistanceDelta(*_box),
        0.5 * (targetBondLengthSquared - distanceSquared)
    );
}

/**
 * @brief tests application of shake algorithm to bond constraint
 */
TEST_F(TestBondConstraint, applyShake)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    const auto delta = _bondConstraint->calculateDistanceDelta(*_box);

    const auto shakeForce = delta.numerical_value_in(DistanceSquared::unit) /
                            (1.0 + 0.5) /
                            normSquared(linalg::Vec3D(0.0, -1.0, -2.0));
    const auto dPos = shakeForce * linalg::Vec3D(0.0, -1.0, -2.0) * angstrom;
    const auto timestep = 2.0 * fs;

    EXPECT_FALSE(_bondConstraint->applyShake(*_box, 0.0, timestep));

    const auto molecule = _box->getMolecules()[0];
    EXPECT_EQ(
        molecule.getAtomPosition(AtomIndex{0}),
        linalg::Vec3D(1.0, 1.0, 1.0) * angstrom + dPos
    );
    EXPECT_EQ(
        molecule.getAtomPosition(AtomIndex{1}),
        linalg::Vec3D(1.0, 2.0, 3.0) * angstrom - 0.5 * dPos
    );

    const auto expectedDeltaVel = (dPos / timestep).in(angstrom / s);
    const auto expectedVel1 =
        linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s - 0.5 * expectedDeltaVel;

    const auto vel0 = molecule.getAtomVelocity(AtomIndex{0});
    const auto vel1 = molecule.getAtomVelocity(AtomIndex{1});

    // the velocity is computed via a different unit conversion path than the
    // expected value, so the results may differ in the last digits
    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_DOUBLE_EQ(
            vel0[i].numerical_value_in(angstrom / s),
            expectedDeltaVel.numerical_value_in(angstrom / s)[i]
        );
        EXPECT_DOUBLE_EQ(
            vel1[i].numerical_value_in(angstrom / s),
            expectedVel1.numerical_value_in(angstrom / s)[i]
        );
    }

    EXPECT_TRUE(_bondConstraint->applyShake(*_box, 1000.0, timestep));
}

/**
 * @brief tests that applyShake rejects a zero-length reference bond vector
 * instead of silently dividing by zero
 */
TEST_F(TestBondConstraint, applyShakeRejectsZeroLengthReference)
{
    _bondConstraint->setShakeDistanceRef(Pos3D{Pos::zero()});

    EXPECT_THROW_MSG(
        (void) _bondConstraint->applyShake(*_box, 0.0, 0.5 * fs),
        exc::ShakeException,
        "Degenerate shake reference bond vector - the reference distance is "
        "zero-length or non-finite, the simulation has become unstable"
    );
}

/**
 * @brief tests that applyShake rejects a non-finite reference bond vector
 * instead of silently propagating NaN forces
 */
TEST_F(TestBondConstraint, applyShakeRejectsNonFiniteReference)
{
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    _bondConstraint->setShakeDistanceRef(
        Pos3D{nan * angstrom, 0.0 * angstrom, 0.0 * angstrom}
    );

    EXPECT_THROW_MSG(
        (void) _bondConstraint->applyShake(*_box, 0.0, 0.5 * fs),
        exc::ShakeException,
        "Degenerate shake reference bond vector - the reference distance is "
        "zero-length or non-finite, the simulation has become unstable"
    );
}

/**
 * @brief tests calculation of bond constraint delta velocity
 *
 */
TEST_F(TestBondConstraint, calculateVelocityDelta)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    const auto scalarProduct =
        dot(linalg::Vec3D(-1.0, -1.0, -1.0), linalg::Vec3D(0.0, -1.0, -2.0));
    EXPECT_EQ(
        _bondConstraint->calculateVelocityDelta().numerical_value_in(
            g_per_mol / s
        ),
        -scalarProduct / (1.0 + 0.5) / 5.0
    );
}

/**
 * @brief tests application of rattle algorithm to bond constraint
 */
TEST_F(TestBondConstraint, applyRattle)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    const auto delta = _bondConstraint->calculateVelocityDelta();

    // delta * r has the unit Mass * Pos / Time (momentum like), the velocity
    // change of an atom is this divided by its mass
    const auto deltaMomentum = delta * _bondConstraint->getShakeDistanceRef();

    EXPECT_FALSE(_bondConstraint->applyRattle(MassPerTime::zero()));

    const auto molecule = _box->getMolecules()[0];

    const auto mass0 = molecule.getAtomMass(AtomIndex{0});
    const auto mass1 = molecule.getAtomMass(AtomIndex{1});

    const Vel3D initialVel1  = linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s;
    const Vel3D expectedVel0 = (deltaMomentum / mass0).in(angstrom / s);
    const Vel3D expectedVel1 =
        initialVel1 - Vel3D((deltaMomentum / mass1).in(angstrom / s));

    const auto vel0 = molecule.getAtomVelocity(AtomIndex{0});
    const auto vel1 = molecule.getAtomVelocity(AtomIndex{1});

    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_DOUBLE_EQ(
            vel0[i].numerical_value_in(angstrom / s),
            expectedVel0[i].numerical_value_in(angstrom / s)
        );
        EXPECT_DOUBLE_EQ(
            vel1[i].numerical_value_in(angstrom / s),
            expectedVel1[i].numerical_value_in(angstrom / s)
        );
    }

    EXPECT_TRUE(_bondConstraint->applyRattle(1000.0 * g_per_mol / s));
}
