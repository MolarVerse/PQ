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

#include <gtest/gtest.h>

#include <cmath>
#include <optional>

#include "constant.hpp"
#include "constantDecay.hpp"
#include "expDecay.hpp"
#include "gtest/gtest.h"

using namespace opt;

/* ---------- ConstantLRStrategy ---------- */

TEST(TestConstantLRStrategy, constructorStoresInitialLearningRate)
{
    const auto lr = ConstantLRStrategy(0.1);
    EXPECT_DOUBLE_EQ(lr.getLearningRate(), 0.1);
}

TEST(TestConstantLRStrategy, updateLearningRateIsNoOp)
{
    auto lr = ConstantLRStrategy(0.1);
    lr.updateLearningRate(5u, 100u);
    EXPECT_DOUBLE_EQ(lr.getLearningRate(), 0.1);
    lr.updateLearningRate(99u, 100u);
    EXPECT_DOUBLE_EQ(lr.getLearningRate(), 0.1);
}

TEST(TestConstantLRStrategy, cloneProducesEquivalentObject)
{
    const auto lr     = ConstantLRStrategy(0.42);
    const auto cloned = lr.clone();
    EXPECT_DOUBLE_EQ(cloned->getLearningRate(), 0.42);
}

/* ---------- ConstantDecayLRStrategy ---------- */

TEST(TestConstantDecayLRStrategy, decaysOnFrequencyHit)
{
    auto lr = ConstantDecayLRStrategy(1.0, 0.1, 2u);
    // step 1: not a multiple of frequency (2), no decay
    lr.updateLearningRate(1u, 100u);
    EXPECT_DOUBLE_EQ(lr.getLearningRate(), 1.0);
    // step 2: hits frequency, applies one decay
    lr.updateLearningRate(2u, 100u);
    EXPECT_DOUBLE_EQ(lr.getLearningRate(), 0.9);
    // step 4: hits frequency again
    lr.updateLearningRate(4u, 100u);
    EXPECT_DOUBLE_EQ(lr.getLearningRate(), 0.8);
}

TEST(TestConstantDecayLRStrategy, cloneProducesEquivalentObject)
{
    const auto lr     = ConstantDecayLRStrategy(0.5, 0.05, 1u);
    const auto cloned = lr.clone();
    EXPECT_DOUBLE_EQ(cloned->getLearningRate(), 0.5);
}

/* ---------- ExpDecayLR ---------- */

TEST(TestExpDecayLR, matchesAnalyticalExpDecayFormula)
{
    const auto initial = 1.0;
    const auto decay   = 0.5;
    const auto nEpochs = 100u;

    auto lr = ExpDecayLR(initial, decay, 1u);

    // After step k: learningRate = initial * exp(-decay * k / nEpochs).
    for (auto step : {1u, 10u, 50u, 100u})
    {
        lr.updateLearningRate(step, nEpochs);
        const auto expected = initial * std::exp(
                                            -decay * static_cast<double>(step) /
                                            static_cast<double>(nEpochs)
                                        );
        EXPECT_DOUBLE_EQ(lr.getLearningRate(), expected);
    }
}

TEST(TestExpDecayLR, learningRateMonotonicallyDecreasesWithStep)
{
    auto       lr      = ExpDecayLR(1.0, 1.0, 1u);
    const auto nEpochs = 100u;

    lr.updateLearningRate(1u, nEpochs);
    const auto lrAt1 = lr.getLearningRate();
    lr.updateLearningRate(50u, nEpochs);
    const auto lrAt50 = lr.getLearningRate();
    lr.updateLearningRate(100u, nEpochs);
    const auto lrAt100 = lr.getLearningRate();

    EXPECT_LT(lrAt50, lrAt1);
    EXPECT_LT(lrAt100, lrAt50);
}

/* ---------- LearningRateStrategy::checkLearningRate (base class) ---------- */

TEST(TestLearningRateStrategy, clampsToMaxAndAppendsWarning)
{
    auto lr = ConstantDecayLRStrategy(
        1.0,
        -10.0,
        1u
    );   // negative "decay" → increase
    lr.setMaxLearningRate(std::optional<double>{1.5});

    // Step 1 would bring learning rate to 11.0; checkLearningRate clamps
    // to 1.5.
    lr.updateLearningRate(1u, 100u);
    EXPECT_DOUBLE_EQ(lr.getLearningRate(), 1.5);
    EXPECT_FALSE(lr.getWarningMessages().empty());
}

TEST(TestLearningRateStrategy, clampsToMinAndAppendsWarning)
{
    auto lr =
        ConstantDecayLRStrategy(0.1, 0.5, 1u);   // decay larger than initial
    lr.setMinLearningRate(0.05);

    // Step 1 would bring learning rate to -0.4; checkLearningRate clamps to
    // 0.05.
    lr.updateLearningRate(1u, 100u);
    EXPECT_DOUBLE_EQ(lr.getLearningRate(), 0.05);
    EXPECT_FALSE(lr.getWarningMessages().empty());
}

TEST(TestLearningRateStrategy, errorMessagesEmptyByDefault)
{
    const auto lr = ConstantLRStrategy(0.1);
    EXPECT_TRUE(lr.getErrorMessages().empty());
}
