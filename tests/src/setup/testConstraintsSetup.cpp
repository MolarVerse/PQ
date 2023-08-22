#include "constants.hpp"
#include "constraintsSetup.hpp"
#include "exceptions.hpp"
#include "testSetup.hpp"

using namespace setup;
using namespace customException;

/**
 * @brief tests setupConstraints function for tolerances
 *
 */
TEST_F(TestSetup, setupConstraintTolerances)
{
    _engine.getSettings().setShakeTolerance(1e-6);
    _engine.getSettings().setRattleTolerance(1e-6);

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

    _engine.getSettings().setShakeMaxIter(100);
    _engine.getSettings().setRattleMaxIter(100);

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
    _engine.getConstraints().activate();
    ConstraintsSetup constraintsSetup(_engine);

    EXPECT_NO_THROW(setupConstraints(_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}