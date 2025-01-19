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

#include <vector>   // for vector

#include "constants/conversionFactors.hpp"           // for _FS_TO_S_
#include "constants/internalConversionFactors.hpp"   // for _V_VERLET_VELOCITY_FACTOR_
#include "gtest/gtest.h"   // for CmpHelperFloatingPointEQ, Message, Test, TestPartResult, EXPECT_EQ, EXPECT_DOUBLE_EQ, EXPECT_TRUE, TestPartResultArray, InitGoogleTest, RUN_ALL_TESTS

/**
 * @brief tests function firstStep of velocity verlet integrator
 *
 */
TEST_F(TestIntegrator, firstStep)
{
    _integrator->firstStep(*_box);

    const auto molecule = _box->getMolecules()[0];
    EXPECT_EQ(molecule.getAtomVelocity(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0));

    auto velocities  = linearAlgebra::Vec3D(1.0, 2.0, 3.0);
    velocities      += 0.1 * linearAlgebra::Vec3D(0.5, 1.5, 2.5) *
                  constants::_V_VERLET_VELOCITY_FACTOR_;

    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[0], velocities[0]);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[1], velocities[1]);
    EXPECT_DOUBLE_EQ(molecule.getAtomVelocity(1)[2], velocities[2]);

    EXPECT_DOUBLE_EQ(molecule.getAtomPosition(0)[0], 0.0);

    EXPECT_DOUBLE_EQ(
        molecule.getAtomPosition(1)[0],
        1.0 + 0.1 * velocities[0] * constants::_FS_TO_S_
    );
    EXPECT_DOUBLE_EQ(
        molecule.getAtomPosition(1)[1],
        1.0 + 0.1 * velocities[1] * constants::_FS_TO_S_
    );
    EXPECT_DOUBLE_EQ(
        molecule.getAtomPosition(1)[2],
        1.0 + 0.1 * velocities[2] * constants::_FS_TO_S_
    );

    EXPECT_EQ(molecule.getAtomForce(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    EXPECT_EQ(molecule.getAtomForce(1), linearAlgebra::Vec3D(0.0, 0.0, 0.0));

    EXPECT_TRUE(
        molecule.getCenterOfMass() != linearAlgebra::Vec3D(0.0, 0.0, 0.0)
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
    EXPECT_EQ(molecule.getAtomVelocity(0), linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(
        molecule.getAtomVelocity(1)[0],
        1.0 + 0.1 * 0.5 * constants::_V_VERLET_VELOCITY_FACTOR_
    );
    EXPECT_DOUBLE_EQ(
        molecule.getAtomVelocity(1)[1],
        2.0 + 0.1 * 1.5 * constants::_V_VERLET_VELOCITY_FACTOR_
    );
    EXPECT_DOUBLE_EQ(
        molecule.getAtomVelocity(1)[2],
        3.0 + 0.1 * 2.5 * constants::_V_VERLET_VELOCITY_FACTOR_
    );
}