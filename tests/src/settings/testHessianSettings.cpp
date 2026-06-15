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

#include "hessianSettings.hpp"

using namespace settings;

TEST(TestHessianSettings, setBuilder)
{
    HessianSettings::setBuilder("central");
    EXPECT_EQ(
        HessianSettings::getBuilder(),
        HessianBuilderType::FINITE_DIFFERENCE_FORCES_CENTRAL
    );

    HessianSettings::setBuilder("forward");
    EXPECT_EQ(
        HessianSettings::getBuilder(),
        HessianBuilderType::FINITE_DIFFERENCE_FORCES_FORWARD
    );

    HessianSettings::setBuilder("five-point");
    EXPECT_EQ(
        HessianSettings::getBuilder(),
        HessianBuilderType::FINITE_DIFFERENCE_FORCES_FIVE_POINT
    );

    HessianSettings::setBuilder("analytic");
    EXPECT_EQ(HessianSettings::getBuilder(), HessianBuilderType::ANALYTIC);

    HessianSettings::setBuilder("unknown");
    EXPECT_EQ(HessianSettings::getBuilder(), HessianBuilderType::NONE);
}

TEST(TestHessianSettings, setFilesAndDisplacement)
{
    HessianSettings::setHessianFile("water.hessian");
    HessianSettings::setHessianInfoFile("water.hessian.info");
    HessianSettings::setDisplacement(0.002);

    EXPECT_EQ(HessianSettings::getHessianFile(), "water.hessian");
    EXPECT_EQ(HessianSettings::getHessianInfoFile(), "water.hessian.info");
    EXPECT_EQ(HessianSettings::getDisplacement(), 0.002);
}

TEST(TestHessianSettings, setOptimizeBeforeHessian)
{
    HessianSettings::setOptimizeBeforeHessian(false);
    EXPECT_FALSE(HessianSettings::optimizeBeforeHessian());

    HessianSettings::setOptimizeBeforeHessian(true);
    EXPECT_TRUE(HessianSettings::optimizeBeforeHessian());
}
