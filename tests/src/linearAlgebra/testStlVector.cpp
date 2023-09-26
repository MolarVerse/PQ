#include "stlVector.hpp"   // for max, mean, sum

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <algorithm>       // for copy
#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), EXPECT_EQ, InitGoogleTest, RUN_ALL_TESTS
#include <vector>          // for allocator, vector

TEST(TestStlVector, sum)
{
    const std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(sum(v1), 10.0);
}

TEST(TestStlVector, mean)
{
    const std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(mean(v1), 2.5);
}

TEST(TestStlVector, max)
{
    const std::vector<double> v1 = {1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(max(v1), 4.0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}