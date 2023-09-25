#include "constants.hpp"

#include <gtest/gtest.h>

/*********************
 * natural constants *
 *********************/

TEST(TestConstants, avogadroNumber) { EXPECT_NEAR(constants::_AVOGADRO_NUMBER_ / 6.02214076e23, 1.0, 1e-9); }

TEST(TestConstants, bohrRadius) { EXPECT_NEAR(constants::_BOHR_RADIUS_ / 5.29177210903e-11, 1.0, 1e-9); }

TEST(TestConstants, planckConstant) { EXPECT_NEAR(constants::_PLANCK_CONSTANT_ / 6.62607015e-34, 1.0, 1e-9); }
TEST(TestConstants, reducedPlanckConstant) { EXPECT_NEAR(constants::_REDUCED_PLANCK_CONSTANT_ / 1.054571817e-34, 1.0, 1e-9); }

TEST(TestConstants, boltzmannConstant) { EXPECT_NEAR(constants::_BOLTZMANN_CONSTANT_ / 1.380649e-23, 1.0, 1e-9); }
TEST(TestConstants, universalGasConstant) { EXPECT_NEAR(constants::_UNIVERSAL_GAS_CONSTANT_ / 8.3144626181532395, 1.0, 1e-9); }

TEST(TestConstants, electronCharge) { EXPECT_NEAR(constants::_ELECTRON_CHARGE_ / 1.602176634e-19, 1.0, 1e-9); }
TEST(TestConstants, electronChargeSquared)
{
    EXPECT_NEAR(constants::_ELECTRON_CHARGE_SQUARED_ / (constants::_ELECTRON_CHARGE_ * constants::_ELECTRON_CHARGE_), 1.0, 1e-9);
}

TEST(TestConstants, electronMass) { EXPECT_NEAR(constants::_ELECTRON_MASS_ / 9.109389754e-31, 1.0, 1e-9); }

TEST(TestConstants, permittivityVacuum) { EXPECT_NEAR(constants::_PERMITTIVITY_VACUUM_ / 8.8541878128e-12, 1.0, 1e-9); }

TEST(TestConstants, speedOfLight) { EXPECT_NEAR(constants::_SPEED_OF_LIGHT_ / 299792458.0, 1.0, 1e-9); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}