#include "angleType.hpp"

#include <gtest/gtest.h>

/**
 * @brief tests operator== for AngleType
 *
 */
TEST(TestAngleType, operatorEqual)
{
    forceField::AngleType angleType1(0, 1.0, 2.0);
    forceField::AngleType angleType2(0, 1.0, 2.0);
    forceField::AngleType angleType3(1, 1.0, 2.0);
    forceField::AngleType angleType4(0, 2.0, 2.0);
    forceField::AngleType angleType5(0, 1.0, 3.0);

    EXPECT_TRUE(angleType1 == angleType2);
    EXPECT_FALSE(angleType1 == angleType3);
    EXPECT_FALSE(angleType1 == angleType4);
    EXPECT_FALSE(angleType1 == angleType5);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}