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

#include "testConstraints.hpp"

#include <gtest/gtest.h>

#include "exceptions.hpp"
#include "gmock/gmock.h"
#include "testSettings.hpp"
#include "throwWithMessage.hpp"
#include "timingsSettings.hpp"

/**
 * @brief tests calculation of all bond constraints ref bond lengths
 *
 */
TEST_F(TestConstraints, calcRefBondLengths)
{
    _constraints->calculateConstraintBondRefs(*_box);
    EXPECT_EQ(
        _constraints->getBondConstraints()[0].getShakeDistanceRef(),
        linalg::Vec3D(0.0, -1.0, -2.0)
    );
    EXPECT_EQ(
        _constraints->getBondConstraints()[1].getShakeDistanceRef(),
        linalg::Vec3D(1.0, -2.0, -3.0)
    );
}

/**
 * @brief test apply shake algorithm to all bond constraints
 *
 */
TEST_F(TestConstraints, applyShakeConverged)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .with(&ConstraintSettings::setShakeMaxIter, 1000)
            .with(&ConstraintSettings::setShakeTolerance, 1.0e-8)
            .apply([](auto &s) { s.activateShake(); })
            .build();
    constraints::Constraints constraints(settings);
    _addDefaultBondConstraints(constraints);
    constraints.calculateConstraintBondRefs(*_box);

    settings::TimingsSettings::setTimeStep(2.0);

    EXPECT_NO_THROW(constraints.applyShake(*_box));

    const auto mol = _box->getMolecules()[0];
    EXPECT_THAT(
        mol.getAtomPosition(AtomIndex{0}),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e-5),
            testing::DoubleNear(1.23167, 1e-5),
            testing::DoubleNear(1.46334, 1e-5)
        )
    );
    EXPECT_THAT(
        mol.getAtomPosition(AtomIndex{1}),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e-5),
            testing::DoubleNear(1.76833, 1e-5),
            testing::DoubleNear(2.53666, 1e-5)
        )
    );

    EXPECT_THAT(
        mol.getAtomVelocity(AtomIndex{0}),
        testing::ElementsAre(
            testing::DoubleNear(0.0, 1e10),
            testing::DoubleNear(1.15836e14, 1e10),
            testing::DoubleNear(2.31672e14, 1e10)
        )
    );
    EXPECT_THAT(
        mol.getAtomVelocity(AtomIndex{1}),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e10),
            testing::DoubleNear(1.0 - 1.15836e14, 1e10),
            testing::DoubleNear(1.0 - 2.31672e14, 1e10)
        )
    );
}

/**
 * @brief test apply shake algorithm to all bond constraints and it does not
 * converge
 *
 */
TEST_F(TestConstraints, applyShakeNotConverged)
{
    const auto settings = testSettings::make<ConstraintSettings>()
                              .with(&ConstraintSettings::setShakeMaxIter, 10)
                              .with(&ConstraintSettings::setShakeTolerance, 0.0)
                              .apply([](auto &s) { s.activateShake(); })
                              .build();
    constraints::Constraints constraints(settings);
    _addDefaultBondConstraints(constraints);
    constraints.calculateConstraintBondRefs(*_box);

    settings::TimingsSettings::setTimeStep(2.0);

    EXPECT_THROW_MSG(
        constraints.applyShake(*_box),
        exc::ShakeException,
        "Shake algorithm did not converge for 2 bonds."
    );
}

/**
 * @brief test apply shake algorithm to all bond constraints and it does not
 * converge but is deactivated
 *
 */
TEST_F(TestConstraints, applyShakeNotConvergedDeactivated)
{
    const auto settings = testSettings::make<ConstraintSettings>()
                              .with(&ConstraintSettings::setShakeMaxIter, 10)
                              .with(&ConstraintSettings::setShakeTolerance, 0.0)
                              .build();
    _constraints = std::make_unique<constraints::Constraints>(settings);
    _addDefaultBondConstraints(*_constraints);
    _constraints->calculateConstraintBondRefs(*_box);

    settings::TimingsSettings::setTimeStep(2.0);

    EXPECT_NO_THROW(_constraints->applyShake(*_box));
}

/**
 * @brief test apply rattle algorithm to all bond constraints
 *
 */
TEST_F(TestConstraints, applyRattleConverged)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .with(&ConstraintSettings::setRattleMaxIter, 1000)
            .with(&ConstraintSettings::setRattleTolerance, 1.0e-4)
            .apply([](auto &s) { s.activateShake(); })
            .build();
    _constraints = std::make_unique<constraints::Constraints>(settings);
    _addDefaultBondConstraints(*_constraints);
    _constraints->calculateConstraintBondRefs(*_box);

    EXPECT_NO_THROW(_constraints->applyRattle(*_box));

    const auto mol = _box->getMolecules()[0];
    EXPECT_THAT(
        mol.getAtomVelocity(AtomIndex{0}),
        testing::ElementsAre(
            testing::DoubleNear(0.0, 1e-5),
            testing::DoubleNear(0.3, 1e-5),
            testing::DoubleNear(0.6, 1e-5)
        )
    );
    EXPECT_THAT(
        mol.getAtomVelocity(AtomIndex{1}),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e-5),
            testing::DoubleNear(0.7, 1e-5),
            testing::DoubleNear(0.4, 1e-5)
        )
    );
}

/**
 * @brief test apply rattle algorithm to all bond constraints and it does not
 * converge
 *
 */
TEST_F(TestConstraints, applyRattleNotConverged)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .with(&ConstraintSettings::setRattleMaxIter, 10)
            .with(&ConstraintSettings::setRattleTolerance, 0.0)
            .apply([](auto &s) { s.activateShake(); })
            .build();
    _constraints = std::make_unique<constraints::Constraints>(settings);
    _addDefaultBondConstraints(*_constraints);
    _constraints->calculateConstraintBondRefs(*_box);

    EXPECT_THROW_MSG(
        _constraints->applyRattle(*_box),
        exc::ShakeException,
        "Rattle algorithm did not converge for 2 bonds."
    );
}

/**
 * @brief test apply rattle algorithm to all bond constraints and it does not
 * converge but is deactivated
 *
 */
TEST_F(TestConstraints, applyRattleNotConvergedDeactivated)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .with(&ConstraintSettings::setRattleMaxIter, 10)
            .with(&ConstraintSettings::setRattleTolerance, 0.0)
            .build();
    _constraints = std::make_unique<constraints::Constraints>(settings);
    _addDefaultBondConstraints(*_constraints);
    _constraints->calculateConstraintBondRefs(*_box);

    EXPECT_NO_THROW(_constraints->applyRattle(*_box));
}
