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

#include <optional>

#include "optimizerSettings.hpp"
#include "testSettings.hpp"

TEST(TestOptimizerSettings, MaxLearningRateIsOptional)
{
    const OptimizerSettings settings;

    EXPECT_FALSE(settings.isMaxLearningRateSet());
    EXPECT_EQ(std::nullopt, settings.getMaxLearningRateIfSet());
}

TEST(TestOptimizerSettings, MaxLearningRateIfSetReturnsTheValue)
{
    const auto settings = testSettings::make<OptimizerSettings>()
                              .with(&OptimizerSettings::setMaxLearningRate, 0.5)
                              .build();

    EXPECT_TRUE(settings.isMaxLearningRateSet());
    EXPECT_EQ(std::optional<double>{0.5}, settings.getMaxLearningRateIfSet());
}
