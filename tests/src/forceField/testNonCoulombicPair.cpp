#include "nonCoulombPair.hpp"

#include <gtest/gtest.h>

/**
 * @brief tests operator== for LennardJonesPair
 *
 */
TEST(TestNonCoulombicPair, operatorEqual_LennardJones)
{
    forceField::LennardJonesPair lennardJonesPair1(0, 1, 1.0, 2.0, 3.0);
    forceField::LennardJonesPair lennardJonesPair2(0, 1, 1.0, 2.0, 3.0);
    forceField::LennardJonesPair lennardJonesPair3(1, 1, 1.0, 2.0, 3.0);
    forceField::LennardJonesPair lennardJonesPair4(0, 2, 1.0, 2.0, 3.0);
    forceField::LennardJonesPair lennardJonesPair5(0, 1, 2.0, 2.0, 3.0);
    forceField::LennardJonesPair lennardJonesPair6(0, 1, 1.0, 3.0, 3.0);
    forceField::LennardJonesPair lennardJonesPair7(0, 1, 1.0, 2.0, 4.0);

    forceField::LennardJonesPair lennardJonesPairChangeOrder(1, 0, 1.0, 2.0, 3.0);

    EXPECT_TRUE(lennardJonesPair1 == lennardJonesPair2);
    EXPECT_FALSE(lennardJonesPair1 == lennardJonesPair3);
    EXPECT_FALSE(lennardJonesPair1 == lennardJonesPair4);
    EXPECT_FALSE(lennardJonesPair1 == lennardJonesPair5);
    EXPECT_FALSE(lennardJonesPair1 == lennardJonesPair6);
    EXPECT_FALSE(lennardJonesPair1 == lennardJonesPair7);
    EXPECT_TRUE(lennardJonesPair1 == lennardJonesPairChangeOrder);
}

/**
 * @brief tests operator== for BuckinghamPair
 *
 */
TEST(TestNonCoulombicPair, operatorEqual_Buckingham)
{
    forceField::BuckinghamPair buckinghamPair1(0, 1, 1.0, 2.0, 3.0, 4.0);
    forceField::BuckinghamPair buckinghamPair2(0, 1, 1.0, 2.0, 3.0, 4.0);
    forceField::BuckinghamPair buckinghamPair3(1, 1, 1.0, 2.0, 3.0, 4.0);
    forceField::BuckinghamPair buckinghamPair4(0, 2, 1.0, 2.0, 3.0, 4.0);
    forceField::BuckinghamPair buckinghamPair5(0, 1, 2.0, 2.0, 3.0, 4.0);
    forceField::BuckinghamPair buckinghamPair6(0, 1, 1.0, 3.0, 3.0, 4.0);
    forceField::BuckinghamPair buckinghamPair7(0, 1, 1.0, 2.0, 4.0, 4.0);
    forceField::BuckinghamPair buckinghamPair8(0, 1, 1.0, 2.0, 3.0, 5.0);

    forceField::BuckinghamPair buckinghamPairChangeOrder(1, 0, 1.0, 2.0, 3.0, 4.0);

    EXPECT_TRUE(buckinghamPair1 == buckinghamPair2);
    EXPECT_FALSE(buckinghamPair1 == buckinghamPair3);
    EXPECT_FALSE(buckinghamPair1 == buckinghamPair4);
    EXPECT_FALSE(buckinghamPair1 == buckinghamPair5);
    EXPECT_FALSE(buckinghamPair1 == buckinghamPair6);
    EXPECT_FALSE(buckinghamPair1 == buckinghamPair7);
    EXPECT_FALSE(buckinghamPair1 == buckinghamPair8);
    EXPECT_TRUE(buckinghamPair1 == buckinghamPairChangeOrder);
}

/**
 * @brief tests operator== for MorsePair
 *
 */
TEST(TestNonCoulombicPair, operatorEqual_Morse)
{
    forceField::MorsePair morsePair1(0, 1, 1.0, 2.0, 3.0, 4.0);
    forceField::MorsePair morsePair2(0, 1, 1.0, 2.0, 3.0, 4.0);
    forceField::MorsePair morsePair3(1, 1, 1.0, 2.0, 3.0, 4.0);
    forceField::MorsePair morsePair4(0, 2, 1.0, 2.0, 3.0, 4.0);
    forceField::MorsePair morsePair5(0, 1, 2.0, 2.0, 3.0, 4.0);
    forceField::MorsePair morsePair6(0, 1, 1.0, 3.0, 3.0, 4.0);
    forceField::MorsePair morsePair7(0, 1, 1.0, 2.0, 4.0, 4.0);
    forceField::MorsePair morsePair8(0, 1, 1.0, 2.0, 3.0, 5.0);

    forceField::MorsePair morsePairChangeOrder(1, 0, 1.0, 2.0, 3.0, 4.0);

    EXPECT_TRUE(morsePair1 == morsePair2);
    EXPECT_FALSE(morsePair1 == morsePair3);
    EXPECT_FALSE(morsePair1 == morsePair4);
    EXPECT_FALSE(morsePair1 == morsePair5);
    EXPECT_FALSE(morsePair1 == morsePair6);
    EXPECT_FALSE(morsePair1 == morsePair7);
    EXPECT_FALSE(morsePair1 == morsePair8);
    EXPECT_TRUE(morsePair1 == morsePairChangeOrder);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
