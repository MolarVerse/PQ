#include "potentialSettings.hpp"

#include <gtest/gtest.h>

/**
 * @brief tests string function of enum nonCoulombType
 *
 */
TEST(TestPotentialSettings, string_nonCoulombType)
{
    EXPECT_EQ(settings::string(settings::NonCoulombType::LJ), "lj");
    EXPECT_EQ(settings::string(settings::NonCoulombType::LJ_9_12), "lj_9_12");
    EXPECT_EQ(settings::string(settings::NonCoulombType::BUCKINGHAM), "buck");
    EXPECT_EQ(settings::string(settings::NonCoulombType::MORSE), "morse");
    EXPECT_EQ(settings::string(settings::NonCoulombType::GUFF), "guff");
    EXPECT_EQ(settings::string(settings::NonCoulombType::NONE), "none");
}

/**
 * @brief tests setNonCoulombType function
 *
 */
TEST(TestPotentialSettings, setNonCoulombType)
{
    settings::PotentialSettings::setNonCoulombType("lj");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::LJ);
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombTypeString(), "lj");

    settings::PotentialSettings::setNonCoulombType("lj_9_12");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::LJ_9_12);
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombTypeString(), "lj_9_12");

    settings::PotentialSettings::setNonCoulombType("buck");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::BUCKINGHAM);
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombTypeString(), "buck");

    settings::PotentialSettings::setNonCoulombType("morse");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::MORSE);
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombTypeString(), "morse");

    settings::PotentialSettings::setNonCoulombType("guff");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::GUFF);
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombTypeString(), "guff");

    settings::PotentialSettings::setNonCoulombType("none");
    EXPECT_EQ(settings::PotentialSettings::getNonCoulombType(), settings::NonCoulombType::NONE);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}