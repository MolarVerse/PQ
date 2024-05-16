#include "potential.hpp"   // for Potential

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, EXPECT_EQ, TestInfo (pt...
#include <memory>          // for allocator

using namespace potential;




/*
* main function to run all tests
*/
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

