#include "testParameterFileReader.hpp"

#include "exceptions.hpp"

using namespace ::testing;

/**
 * @brief tests isNeeded function
 *
 * @return true if forceField is enabled
 * @return false
 */
TEST_F(TestParameterFileReader, isNeeded)
{
    EXPECT_FALSE(_parameterFileReader->isNeeded());

    _engine->getForceField().activate();
    EXPECT_TRUE(_parameterFileReader->isNeeded());
}

TEST_F(TestParameterFileReader, determineSection) {}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}