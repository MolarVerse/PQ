#include "dihedralType.hpp"   // for DihedralType

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult
#include <gtest/gtest.h>   // for Test, EXPECT_FALSE, InitGoogleTest, RUN_...
#include <string>          // for allocator, string

/**
 * @brief tests operator== for DihedralType
 *
 */
TEST(TestDihedralType, operatorEqual)
{
    forceField::DihedralType dihedralType1(0, 1.0, 2.0, 3.0);
    forceField::DihedralType dihedralType2(0, 1.0, 2.0, 3.0);
    forceField::DihedralType dihedralType3(1, 1.0, 2.0, 3.0);
    forceField::DihedralType dihedralType4(0, 2.0, 2.0, 3.0);
    forceField::DihedralType dihedralType5(0, 1.0, 3.0, 3.0);
    forceField::DihedralType dihedralType6(0, 1.0, 2.0, 4.0);

    EXPECT_TRUE(dihedralType1 == dihedralType2);
    EXPECT_FALSE(dihedralType1 == dihedralType3);
    EXPECT_FALSE(dihedralType1 == dihedralType4);
    EXPECT_FALSE(dihedralType1 == dihedralType5);
    EXPECT_FALSE(dihedralType1 == dihedralType6);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}