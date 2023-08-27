#include "bondType.hpp"   // for BondType

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult
#include <gtest/gtest.h>   // for Test, EXPECT_FALSE, InitGoogleTest, RUN_ALL...
#include <string>          // for allocator, string

/**
 * @brief tests operator== for BondType
 *
 */
TEST(TestBondType, operatorEqual)
{
    forceField::BondType bondType1(0, 1.0, 2.0);
    forceField::BondType bondType2(0, 1.0, 2.0);
    forceField::BondType bondType3(1, 1.0, 2.0);
    forceField::BondType bondType4(0, 2.0, 2.0);
    forceField::BondType bondType5(0, 1.0, 3.0);

    EXPECT_TRUE(bondType1 == bondType2);
    EXPECT_FALSE(bondType1 == bondType3);
    EXPECT_FALSE(bondType1 == bondType4);
    EXPECT_FALSE(bondType1 == bondType5);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}