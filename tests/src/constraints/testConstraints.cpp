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
#include "unit_types.hpp"
#include "units.hpp"

/**
 * @brief tests calculation of all bond constraints ref bond lengths
 *
 */
TEST_F(TestConstraints, calcRefBondLengths)
{
    _constraints->calculateConstraintBondRefs(*_box);
    EXPECT_EQ(
        _constraints->getBondConstraints()[0].getShakeDistanceRef(),
        linalg::Vec3D(0.0, -1.0, -2.0) * angstrom
    );
    EXPECT_EQ(
        _constraints->getBondConstraints()[1].getShakeDistanceRef(),
        linalg::Vec3D(1.0, -2.0, -3.0) * angstrom
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
            .with(&ConstraintSettings::setShakeMaxIterByUser, 1000)
            .with(&ConstraintSettings::setShakeToleranceByUser, 1.0e-8)
            .apply([](auto &config) { config.activateShakeByUser(); })
            .build();
    constraints::Constraints constraints(settings);
    _addDefaultBondConstraints(constraints);
    constraints.calculateConstraintBondRefs(*_box);

    EXPECT_NO_THROW(constraints.applyShake(*_box, 2.0 * fs));

    const auto molecule = _box->getMolecules()[0];
    EXPECT_THAT(
        molecule.getAtomPosition(AtomIndex{0}).numerical_value_in(angstrom),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e-5),
            testing::DoubleNear(1.23167, 1e-5),
            testing::DoubleNear(1.46334, 1e-5)
        )
    );
    EXPECT_THAT(
        molecule.getAtomPosition(AtomIndex{1}).numerical_value_in(angstrom),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e-5),
            testing::DoubleNear(1.76833, 1e-5),
            testing::DoubleNear(2.53666, 1e-5)
        )
    );

    EXPECT_THAT(
        molecule.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s),
        testing::ElementsAre(
            testing::DoubleNear(0.0, 1e10),
            testing::DoubleNear(1.15836e14, 1e10),
            testing::DoubleNear(2.31672e14, 1e10)
        )
    );
    EXPECT_THAT(
        molecule.getAtomVelocity(AtomIndex{1}).numerical_value_in(angstrom / s),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e10),
            testing::DoubleNear(1.0 - 1.15836e14, 1e10),
            testing::DoubleNear(1.0 - 2.31672e14, 1e10)
        )
    );
}

TEST_F(TestConstraints, applyShakeToPositionsKeepsVelocities)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .apply([](auto &config) { config.activateShakeByUser(); })
            .build();
    constraints::Constraints constraints(settings);

    auto simulationBox = _makeDimerBox();
    auto *molecule     = simulationBox->getMolecules().data();
    constraints.addBondConstraint(
        constraints::BondConstraint(
            molecule,
            molecule,
            AtomIndex{0},
            AtomIndex{1},
            1.0 * angstrom
        )
    );
    constraints.calculateConstraintBondRefs(*simulationBox);

    constraints.applyShakeToPositions(*simulationBox);

    const auto distance = norm(
        molecule->getAtomPosition(AtomIndex{1}) -
        molecule->getAtomPosition(AtomIndex{0})
    );
    EXPECT_NEAR(distance.numerical_value_in(angstrom), 1.0, 1e-7);

    for (const auto &atom : simulationBox->getAtoms())
    {
        EXPECT_THAT(
            atom->getVelocity().numerical_value_in(angstrom / s),
            testing::ElementsAre(5.0, 5.0, 5.0)
        );
    }
}

TEST_F(TestConstraints, projectForcesRemovesTheBondComponent)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .apply([](auto &config) { config.activateShakeByUser(); })
            .build();
    constraints::Constraints constraints(settings);

    auto simulationBox = _makeDimerBox();
    auto *molecule     = simulationBox->getMolecules().data();
    constraints.addBondConstraint(
        constraints::BondConstraint(
            molecule,
            molecule,
            AtomIndex{0},
            AtomIndex{1},
            1.1 * angstrom
        )
    );
    constraints.calculateConstraintBondRefs(*simulationBox);

    auto &atoms = simulationBox->getAtoms();
    atoms[0]->setForce(linalg::Vec3D(4.0, 2.0, 0.0) * kcal_per_mol / angstrom);
    atoms[1]->setForce(linalg::Vec3D(0.0, 0.0, 0.0) * kcal_per_mol / angstrom);

    constraints.projectForces(*simulationBox);

    // the bond direction keeps the force that moves both atoms together
    EXPECT_THAT(
        atoms[0]->getForce().numerical_value_in(kcal_per_mol / angstrom),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e-9),
            testing::DoubleNear(2.0, 1e-9),
            testing::DoubleNear(0.0, 1e-9)
        )
    );
    EXPECT_THAT(
        atoms[1]->getForce().numerical_value_in(kcal_per_mol / angstrom),
        testing::ElementsAre(
            testing::DoubleNear(3.0, 1e-9),
            testing::DoubleNear(0.0, 1e-9),
            testing::DoubleNear(0.0, 1e-9)
        )
    );

    for (const auto &atom : atoms)
    {
        EXPECT_THAT(
            atom->getVelocity().numerical_value_in(angstrom / s),
            testing::ElementsAre(5.0, 5.0, 5.0)
        );
    }
}

TEST_F(TestConstraints, projectForcesWithoutShakeKeepsTheForces)
{
    const auto settings = testSettings::make<ConstraintSettings>().build();
    constraints::Constraints constraints(settings);

    auto simulationBox = _makeDimerBox();
    auto &atoms        = simulationBox->getAtoms();
    atoms[0]->setForce(linalg::Vec3D(4.0, 2.0, 0.0) * kcal_per_mol / angstrom);

    constraints.projectForces(*simulationBox);
    constraints.applyShakeToPositions(*simulationBox);

    EXPECT_THAT(
        atoms[0]->getForce().numerical_value_in(kcal_per_mol / angstrom),
        testing::ElementsAre(4.0, 2.0, 0.0)
    );
    EXPECT_THAT(
        atoms[1]->getPosition().numerical_value_in(angstrom),
        testing::ElementsAre(2.1, 1.0, 1.0)
    );
}

/**
 * @brief test apply shake algorithm to all bond constraints and it does not
 * converge
 *
 */
TEST_F(TestConstraints, applyShakeNotConverged)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .with(&ConstraintSettings::setShakeMaxIterByUser, 10)
            .with(&ConstraintSettings::setShakeToleranceByUser, 0.0)
            .apply([](auto &config) { config.activateShakeByUser(); })
            .build();
    constraints::Constraints constraints(settings);
    _addDefaultBondConstraints(constraints);
    constraints.calculateConstraintBondRefs(*_box);

    EXPECT_THROW_MSG(
        constraints.applyShake(*_box, 2.0 * fs),
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
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .with(&ConstraintSettings::setShakeMaxIterByUser, 10)
            .with(&ConstraintSettings::setShakeToleranceByUser, 0.0)
            .build();
    _constraints = std::make_unique<constraints::Constraints>(settings);
    _addDefaultBondConstraints(*_constraints);
    _constraints->calculateConstraintBondRefs(*_box);

    EXPECT_NO_THROW(_constraints->applyShake(*_box, 2.0 * fs));
}

/**
 * @brief test apply rattle algorithm to all bond constraints
 *
 */
TEST_F(TestConstraints, applyRattleConverged)
{
    const auto settings =
        testSettings::make<ConstraintSettings>()
            .with(&ConstraintSettings::setRattleMaxIterByUser, 1000)
            .with(
                &ConstraintSettings::setRattleToleranceByUser,
                1.0e-4 * g_per_mol / s
            )
            .apply([](auto &config) { config.activateShakeByUser(); })
            .build();
    _constraints = std::make_unique<constraints::Constraints>(settings);
    _addDefaultBondConstraints(*_constraints);
    _constraints->calculateConstraintBondRefs(*_box);

    EXPECT_NO_THROW(_constraints->applyRattle(*_box));

    const auto molecule = _box->getMolecules()[0];
    EXPECT_THAT(
        molecule.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s),
        testing::ElementsAre(
            testing::DoubleNear(0.0, 1e-5),
            testing::DoubleNear(0.3, 1e-5),
            testing::DoubleNear(0.6, 1e-5)
        )
    );
    EXPECT_THAT(
        molecule.getAtomVelocity(AtomIndex{1}).numerical_value_in(angstrom / s),
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
            .with(&ConstraintSettings::setRattleMaxIterByUser, 10)
            .with(
                &ConstraintSettings::setRattleToleranceByUser,
                MassPerTime::zero()
            )
            .apply([](auto &config) { config.activateShakeByUser(); })
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
            .with(&ConstraintSettings::setRattleMaxIterByUser, 10)
            .with(
                &ConstraintSettings::setRattleToleranceByUser,
                MassPerTime::zero()
            )
            .build();
    _constraints = std::make_unique<constraints::Constraints>(settings);
    _addDefaultBondConstraints(*_constraints);
    _constraints->calculateConstraintBondRefs(*_box);

    EXPECT_NO_THROW(_constraints->applyRattle(*_box));
}
