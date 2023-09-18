#include "qmSettings.hpp"   // for QMSettings, QMMethod

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, InitGoogleTest, RUN_ALL_TESTS, EXPECT_EQ
#include <memory>          // for allocator

TEST(QMSettingsTest, SetQMMethodTest)
{
    settings::QMSettings::setQMMethod("dftbplus");
    EXPECT_EQ(settings::QMSettings::getQMMethod(), settings::QMMethod::DFTBPLUS);

    settings::QMSettings::setQMMethod("none");
    EXPECT_EQ(settings::QMSettings::getQMMethod(), settings::QMMethod::NONE);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}