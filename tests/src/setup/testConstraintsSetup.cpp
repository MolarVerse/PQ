#include "constraintSettings.hpp"   // for getShakeMaxIter, getShakeTolerance, getRattleMaxIter, getRattleTolerance
#include "constraints.hpp"          // for Constraints
#include "constraintsSetup.hpp"     // for ConstraintsSetup, setupConstraints
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for customException
#include "settings.hpp"             // for Settings
#include "testSetup.hpp"            // for TestSetup

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for TestInfo (ptr only), EXPECT_EQ, Init...
#include <string>          // for allocator, basic_string

using namespace setup;
using namespace customException;

/**
 * @brief tests setupConstraints function for tolerances
 *
 */
TEST_F(TestSetup, setupConstraintTolerances)
{
    settings::ConstraintSettings::setShakeTolerance(1e-6);
    settings::ConstraintSettings::setRattleTolerance(1e-6);

    _engine.getConstraints().activate();

    ConstraintsSetup constraintsSetup(_engine);
    constraintsSetup.setup();

    EXPECT_EQ(_engine.getConstraints().getShakeTolerance(), 1e-6);
    EXPECT_EQ(_engine.getConstraints().getRattleTolerance(), 1e-6);
}

/**
 * @brief tests setupConstraints function for max iterations
 *
 */
TEST_F(TestSetup, setupConstraintMaxIter)
{
    settings::ConstraintSettings::setShakeMaxIter(100);
    settings::ConstraintSettings::setRattleMaxIter(100);

    _engine.getConstraints().activate();

    ConstraintsSetup constraintsSetup(_engine);
    constraintsSetup.setup();

    EXPECT_EQ(_engine.getConstraints().getShakeMaxIter(), 100);
    EXPECT_EQ(_engine.getConstraints().getRattleMaxIter(), 100);
}

/**
 * @brief tests setupConstraints wrapper function - should not throw
 *
 */
TEST_F(TestSetup, setupConstraints)
{
    settings::ConstraintSettings::setShakeTolerance(999.0);

    _engine.getConstraints().deactivate();
    EXPECT_NO_THROW(setupConstraints(_engine));
    const auto shakeToleranceDeactivated = _engine.getConstraints().getShakeTolerance();

    _engine.getConstraints().activate();
    EXPECT_NO_THROW(setupConstraints(_engine));
    const auto shakeToleranceActivated = _engine.getConstraints().getShakeTolerance();

    EXPECT_NE(shakeToleranceDeactivated, shakeToleranceActivated);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}