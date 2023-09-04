#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"

#include <gtest/gtest.h>   // for AssertionResult, Message, TestPartResult

TEST(TestBondForceField, calculateEnergyAndForces) {}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}