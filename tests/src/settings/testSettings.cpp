#include "exceptions.hpp"
#include "settings.hpp"

#include <gtest/gtest.h>

TEST(TestSettings, setTemperature)
{
    auto settings = settings::Settings();
    EXPECT_THROW(settings.setTemperature(-1.0), customException::InputFileException);
}

TEST(TestSettings, setRelaxationTime)
{
    auto settings = settings::Settings();
    EXPECT_THROW(settings.setRelaxationTime(-1.0), customException::InputFileException);
}

TEST(TestSettings, setTauManostat)
{
    auto settings = settings::Settings();
    EXPECT_THROW(settings.setTauManostat(-1.0), customException::InputFileException);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}