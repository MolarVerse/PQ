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

TEST(TestHessianSettings, setFilesAndDisplacement)
{
    settings::HessianSettings::setHessianFile("water.hessian");
    settings::HessianSettings::setHessianInfoFile("water.hessian.info");
    settings::HessianSettings::setDisplacement(0.002);

    EXPECT_EQ(settings::HessianSettings::getHessianFile(), "water.hessian");
    EXPECT_EQ(
        settings::HessianSettings::getHessianInfoFile(),
        "water.hessian.info"
    );
    EXPECT_EQ(settings::HessianSettings::getDisplacement(), 0.002);
}

TEST(TestHessianSettings, setOptimizeBeforeHessian)
{
    settings::HessianSettings::setOptimizeBeforeHessian(false);
    EXPECT_FALSE(settings::HessianSettings::optimizeBeforeHessian());

    settings::HessianSettings::setOptimizeBeforeHessian(true);
    EXPECT_TRUE(settings::HessianSettings::optimizeBeforeHessian());
}
