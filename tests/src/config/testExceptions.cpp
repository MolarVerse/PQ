#include "exceptions.hpp"
#include "throwWithMessage.hpp"

#include <gtest/gtest.h>

TEST(TestExceptions, throwWithMessage) { EXPECT_THROW_MSG(); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}