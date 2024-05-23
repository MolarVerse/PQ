#include <gtest/gtest.h>   // for Test, EXPECT_EQ, TestInfo (pt...

#include <memory>   // for allocator

#include "gtest/gtest.h"   // for Message, TestPartResult
#include "potential.hpp"   // for Potential

using namespace potential;

TEST(TestPotential, placeholder) { 
    EXPECT_TRUE(true); 

}

/*
 * main function to run all tests
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}