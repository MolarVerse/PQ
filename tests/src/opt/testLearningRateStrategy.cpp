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

using namespace opt;

/* ---------- ConstantLRStrategy ---------- */

TEST(TestConstantLRStrategy, constructorStoresInitialLearningRate)
{
    const auto learningRate = ConstantLRStrategy(0.1);
    EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), 0.1);
}

TEST(TestConstantLRStrategy, updateLearningRateIsNoOp)
{
    auto learningRate = ConstantLRStrategy(0.1);
    learningRate.updateLearningRate(5U, 100U);
    EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), 0.1);
    learningRate.updateLearningRate(99U, 100U);
    EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), 0.1);
}

TEST(TestConstantLRStrategy, cloneProducesEquivalentObject)
{
    const auto learningRate = ConstantLRStrategy(0.42);
    const auto cloned       = learningRate.clone();
    EXPECT_DOUBLE_EQ(cloned->getLearningRate(), 0.42);
}

/* ---------- ConstantDecayLRStrategy ---------- */

TEST(TestConstantDecayLRStrategy, decaysOnFrequencyHit)
{
    auto learningRate = ConstantDecayLRStrategy(1.0, 0.1, 2U);
    // step 1: not a multiple of frequency (2), no decay
    learningRate.updateLearningRate(1U, 100U);
    EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), 1.0);
    // step 2: hits frequency, applies one decay
    learningRate.updateLearningRate(2U, 100U);
    EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), 0.9);
    // step 4: hits frequency again
    learningRate.updateLearningRate(4U, 100U);
    EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), 0.8);
}

TEST(TestConstantDecayLRStrategy, cloneProducesEquivalentObject)
{
    const auto learningRate = ConstantDecayLRStrategy(0.5, 0.05, 1U);
    const auto cloned       = learningRate.clone();
    EXPECT_DOUBLE_EQ(cloned->getLearningRate(), 0.5);
}

/* ---------- ExpDecayLR ---------- */

TEST(TestExpDecayLR, matchesAnalyticalExpDecayFormula)
{
    const auto initial = 1.0;
    const auto decay   = 0.5;
    const auto nEpochs = 100U;

    auto learningRate = ExpDecayLR(initial, decay, 1U);

    // After step k: learningRate = initial * exp(-decay * k / nEpochs).
    for (auto step : {1U, 10U, 50U, 100U})
    {
        learningRate.updateLearningRate(step, nEpochs);
        const auto expected = initial * std::exp(
                                            -decay * static_cast<double>(step) /
                                            static_cast<double>(nEpochs)
                                        );
        EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), expected);
    }
}

TEST(TestExpDecayLR, learningRateMonotonicallyDecreasesWithStep)
{
    auto       learningRate = ExpDecayLR(1.0, 1.0, 1U);
    const auto nEpochs      = 100U;

    learningRate.updateLearningRate(1U, nEpochs);
    const auto lrAt1 = learningRate.getLearningRate();
    learningRate.updateLearningRate(50U, nEpochs);
    const auto lrAt50 = learningRate.getLearningRate();
    learningRate.updateLearningRate(100U, nEpochs);
    const auto lrAt100 = learningRate.getLearningRate();

    EXPECT_LT(lrAt50, lrAt1);
    EXPECT_LT(lrAt100, lrAt50);
}

/* ---------- LearningRateStrategy::checkLearningRate (base class) ---------- */

TEST(TestLearningRateStrategy, clampsToMaxAndAppendsWarning)
{
    auto learningRate = ConstantDecayLRStrategy(
        1.0,
        -10.0,
        1U
    );   // negative "decay" → increase
    learningRate.setMaxLearningRate(std::optional<double>{1.5});

    // Step 1 would bring learning rate to 11.0; checkLearningRate clamps
    // to 1.5.
    learningRate.updateLearningRate(1U, 100U);
    EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), 1.5);
    EXPECT_FALSE(learningRate.getWarningMessages().empty());
}

TEST(TestLearningRateStrategy, clampsToMinAndAppendsWarning)
{
    auto learningRate =
        ConstantDecayLRStrategy(0.1, 0.5, 1U);   // decay larger than initial
    learningRate.setMinLearningRate(0.05);

    // Step 1 would bring learning rate to -0.4; checkLearningRate clamps to
    // 0.05.
    learningRate.updateLearningRate(1U, 100U);
    EXPECT_DOUBLE_EQ(learningRate.getLearningRate(), 0.05);
    EXPECT_FALSE(learningRate.getWarningMessages().empty());
}

TEST(TestLearningRateStrategy, errorMessagesEmptyByDefault)
{
    const auto learningRate = ConstantLRStrategy(0.1);
    EXPECT_TRUE(learningRate.getErrorMessages().empty());
}
