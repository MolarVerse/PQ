/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#include "timingsSettings.hpp"

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult
#include <string>          // for string

/**
 * @brief tests calculation of bond constraint ref bond length
 *
 */
TEST_F(TestBondConstraint, calcRefBondLength)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    EXPECT_EQ(_bondConstraint->getShakeDistanceRef(), linearAlgebra::Vec3D(0.0, -1.0, -2.0));
}

/**
 * @brief tests calculation of bond constraint delta bond length
 *
 */
TEST_F(TestBondConstraint, calculateDistanceDelta)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    const auto distanceSquared         = normSquared(linearAlgebra::Vec3D(0.0, -1.0, -2.0));
    const auto targetBondLengthSquared = _targetBondLength * _targetBondLength;
    EXPECT_EQ(_bondConstraint->calculateDistanceDelta(*_box), 0.5 * (targetBondLengthSquared - distanceSquared));
}

/**
 * @brief tests application of shake algorithm to bond constraint
 */
TEST_F(TestBondConstraint, applyShake)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    const auto delta = _bondConstraint->calculateDistanceDelta(*_box);

    const auto shakeForce = delta / (1.0 + 0.5) / normSquared(linearAlgebra::Vec3D(0.0, -1.0, -2.0));
    const auto dPos       = shakeForce * linearAlgebra::Vec3D(0.0, -1.0, -2.0);
    const auto timestep   = 2.0;
    settings::TimingsSettings::setTimeStep(timestep);

    EXPECT_FALSE(_bondConstraint->applyShake(*_box, 0.0));

    EXPECT_EQ(_box->getMolecules()[0].getAtomPosition(0), linearAlgebra::Vec3D(1.0, 1.0, 1.0) + dPos);
    EXPECT_EQ(_box->getMolecules()[0].getAtomPosition(1), linearAlgebra::Vec3D(1.0, 2.0, 3.0) - 0.5 * dPos);

    EXPECT_EQ(_box->getMolecules()[0].getAtomVelocity(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0) + dPos / timestep);
    EXPECT_EQ(_box->getMolecules()[0].getAtomVelocity(1), linearAlgebra::Vec3D(1.0, 1.0, 1.0) - 0.5 * dPos / timestep);

    EXPECT_TRUE(_bondConstraint->applyShake(*_box, 1000.0));
}

/**
 * @brief tests calculation of bond constraint delta velocity
 *
 */
TEST_F(TestBondConstraint, calculateVelocityDelta)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    const auto scalarProduct = dot(linearAlgebra::Vec3D(-1.0, -1.0, -1.0), linearAlgebra::Vec3D(0.0, -1.0, -2.0));
    EXPECT_EQ(_bondConstraint->calculateVelocityDelta(), -scalarProduct / (1.0 + 0.5) / 5.0);
}

/**
 * @brief tests application of rattle algorithm to bond constraint
 */
TEST_F(TestBondConstraint, applyRattle)
{
    _bondConstraint->calculateConstraintBondRef(*_box);
    const auto delta = _bondConstraint->calculateVelocityDelta();
    const auto dv    = delta * _bondConstraint->getShakeDistanceRef();

    EXPECT_FALSE(_bondConstraint->applyRattle(0.0));
    EXPECT_EQ(_box->getMolecules()[0].getAtomVelocity(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0) + dv);
    EXPECT_EQ(_box->getMolecules()[0].getAtomVelocity(1), linearAlgebra::Vec3D(1.0, 1.0, 1.0) - 0.5 * dv);

    EXPECT_TRUE(_bondConstraint->applyRattle(1000.0));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}