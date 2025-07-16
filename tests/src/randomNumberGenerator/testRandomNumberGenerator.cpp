/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include <gtest/gtest.h>   // for Test, InitGoogleTest, RUN_ALL_TESTS

#include "gtest/gtest.h"               // for Message, TestPartResult
#include "randomNumberGenerator.hpp"   // for RandomNumberGenerator
#include "settings.hpp"                // for Settings

using namespace randomNumberGenerator;
using namespace settings;

/**
 * @brief test randomNumberGenerator uniform real distribution range
 *
 */
TEST(TestRandomNumberGenerator, uniformRealDistributionRange)
{
    RandomNumberGenerator rng{};
    constexpr double      min{5.0};
    constexpr double      max{10.0};
    constexpr int         numSamples{1000};

    for (int i{0}; i < numSamples; ++i)
    {
        double value = rng.getUniformRealDistribution(min, max);
        EXPECT_GE(value, min);
        EXPECT_LE(value, max);
    }
}

/**
 * @brief test randomNumberGenerator uniform real distribution mean
 *
 */
TEST(TestRandomNumberGenerator, uniformRealDistributionMean)
{
    RandomNumberGenerator rng{};
    constexpr double      min{0.0};
    constexpr double      max{10.0};
    constexpr int         numSamples{10000};

    double sum{0.0};
    for (int i{0}; i < numSamples; ++i)
    {
        sum += rng.getUniformRealDistribution(min, max);
    }

    double actualMean{sum / numSamples};
    double expectedMean{(min + max) / 2};
    EXPECT_NEAR(actualMean, expectedMean, 0.1);
}

/**
 * @brief test randomNumberGenerator normal distribution mean
 *
 */
TEST(TestRandomNumberGenerator, normalDistributionMean)
{
    RandomNumberGenerator rng{};
    constexpr double      mean{5.0};
    constexpr double      stddev{2.0};
    constexpr int         numSamples{10000};

    double sum{0.0};
    for (int i{0}; i < numSamples; ++i)
    {
        sum += rng.getNormalDistribution(mean, stddev);
    }

    double actualMean{sum / numSamples};
    EXPECT_NEAR(actualMean, mean, 0.1);
}

/**
 * @brief test randomNumberGenerator determinism with seed
 *
 */
TEST(TestRandomNumberGenerator, determinismWithSeed)
{
    Settings::setIsRandomSeedSet(true);
    Settings::setRandomSeed(73);

    RandomNumberGenerator rng1{};
    RandomNumberGenerator rng2{};

    for (int i{0}; i < 100; ++i)
    {
        double val1 = rng1.getNormalDistribution(0.0, 1.0);
        double val2 = rng2.getNormalDistribution(0.0, 1.0);
        EXPECT_EQ(val1, val2);
    }
}

/**
 * @brief test randomNumberGenerator edge cases
 *
 */
TEST(TestRandomNumberGenerator, edgeCases)
{
    RandomNumberGenerator rng{};

    EXPECT_NO_THROW(rng.getUniformRealDistribution(5.0, 5.0));
    EXPECT_NO_THROW(rng.getNormalDistribution(0.0, 0.0));
    EXPECT_NO_THROW(rng.getNormalDistribution(-10.0, 1.0));
    EXPECT_NO_THROW(rng.getNormalDistribution(0.0, 0.00001));
}

/**
 * @brief test randomNumberGenerator normal distribution shape
 *
 */
TEST(TestRandomNumberGenerator, normalDistributionShape)
{
    RandomNumberGenerator rng{};

    constexpr double mean{0.0};
    constexpr double stddev{1.0};
    constexpr int    numSamples{10000};

    std::vector<double> samples{};
    samples.reserve(numSamples);

    for (int i{0}; i < numSamples; ++i)
    {
        samples.push_back(rng.getNormalDistribution(mean, stddev));
    }

    int withinOneStddev = 0;
    for (double val : samples)
    {
        if (std::abs(val - mean) <= stddev)
            ++withinOneStddev;
    }

    double percentage = static_cast<double>(withinOneStddev) / numSamples;
    EXPECT_NEAR(percentage, 0.68, 0.03);
}