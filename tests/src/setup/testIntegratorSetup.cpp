#include <gtest/gtest.h>   // for CmpHelperFloatingPointEQ, InitGoogleTest

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}