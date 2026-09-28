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

#include "testDistanceConstraint.hpp"

#include <gtest/gtest.h>

#include "exceptions.hpp"         // for ShakeException
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

/**
 * @brief tests that no force is applied while the distance is within bounds
 *
 */
TEST_F(TestDistanceConstraint, withinBoundsAppliesNoForce)
{
    // atom1 = (1,1,1), atom2 = (1,2,3) -> distance = sqrt(5) ~= 2.236,
    // within [1.0, 3.0]
    _distanceConstraint->applyDistanceConstraint(*_box, 1.0);

    EXPECT_DOUBLE_EQ(_distanceConstraint->getLowerEnergy(), 0.0);
    EXPECT_DOUBLE_EQ(_distanceConstraint->getUpperEnergy(), 0.0);

    const auto mol = _box->getMolecules()[0];
    EXPECT_EQ(mol.getAtomForce(AtomIndex{0}), linalg::Vec3D(0.0, 0.0, 0.0));
    EXPECT_EQ(mol.getAtomForce(AtomIndex{1}), linalg::Vec3D(0.0, 0.0, 0.0));
}

/**
 * @brief tests that a restoring force is applied when the lower bound is
 * violated
 *
 */
TEST_F(TestDistanceConstraint, belowLowerBoundAppliesRestoringForce)
{
    _atom2->setPosition(linalg::Vec3D(1.0, 1.5, 1.0));

    // dPos = (0, 0.5, 0), distance = 0.5, below lowerDistance = 1.0
    _distanceConstraint->applyDistanceConstraint(*_box, 1.0);

    const auto delta          = _lowerDistance - 0.5;
    const auto expectedEnergy = 0.5 * _springConstant * delta * delta;
    EXPECT_DOUBLE_EQ(_distanceConstraint->getLowerEnergy(), expectedEnergy);
    EXPECT_DOUBLE_EQ(_distanceConstraint->getUpperEnergy(), 0.0);

    const auto expectedForce =
        -_springConstant * delta * linalg::Vec3D(0.0, 0.5, 0.0) / 0.5;

    const auto mol = _box->getMolecules()[0];
    EXPECT_EQ(mol.getAtomForce(AtomIndex{0}), expectedForce);
    EXPECT_EQ(mol.getAtomForce(AtomIndex{1}), -expectedForce);
}

/**
 * @brief tests that a restoring force is applied when the upper bound is
 * violated
 *
 */
TEST_F(TestDistanceConstraint, aboveUpperBoundAppliesRestoringForce)
{
    _atom2->setPosition(linalg::Vec3D(1.0, 5.0, 1.0));

    // dPos = (0, 4, 0), distance = 4.0, above upperDistance = 3.0
    _distanceConstraint->applyDistanceConstraint(*_box, 1.0);

    const auto delta          = 4.0 - _upperDistance;
    const auto expectedEnergy = 0.5 * _springConstant * delta * delta;
    EXPECT_DOUBLE_EQ(_distanceConstraint->getUpperEnergy(), expectedEnergy);
    EXPECT_DOUBLE_EQ(_distanceConstraint->getLowerEnergy(), 0.0);

    const auto expectedForce =
        _springConstant * delta * linalg::Vec3D(0.0, 4.0, 0.0) / 4.0;

    const auto mol = _box->getMolecules()[0];
    EXPECT_EQ(mol.getAtomForce(AtomIndex{0}), expectedForce);
    EXPECT_EQ(mol.getAtomForce(AtomIndex{1}), -expectedForce);
}

/**
 * @brief tests that a negative time interval is a no-op
 *
 */
TEST_F(TestDistanceConstraint, negativeTimeIntervalIsNoOp)
{
    _atom2->setPosition(linalg::Vec3D(1.0, 5.0, 1.0));

    EXPECT_NO_THROW(_distanceConstraint->applyDistanceConstraint(*_box, -1.0));

    EXPECT_DOUBLE_EQ(_distanceConstraint->getLowerEnergy(), 0.0);
    EXPECT_DOUBLE_EQ(_distanceConstraint->getUpperEnergy(), 0.0);
}

/**
 * @brief tests that a zero-length, constraint-violating separation is
 * rejected instead of silently producing a NaN force
 *
 */
TEST_F(TestDistanceConstraint, rejectsZeroLengthViolatingSeparation)
{
    // both atoms collapse onto the same position -> distance = 0.0, which
    // is below lowerDistance = 1.0
    _atom2->setPosition(_atom1->getPosition());

    EXPECT_THROW_MSG(
        _distanceConstraint->applyDistanceConstraint(*_box, 1.0),
        exc::ShakeException,
        "Degenerate distance-constraint separation - the atom separation is "
        "zero-length or non-finite, the simulation has become unstable"
    );
}
