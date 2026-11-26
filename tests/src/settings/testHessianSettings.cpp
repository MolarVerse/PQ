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
#include "units.hpp"

TEST(TestHessianSettings, setFilesAndDisplacement)
{
    HessianSettings hessianSettings;
    hessianSettings.set_hessianFileByUser("water.hessian");
    hessianSettings.set_hessianInfoFileByUser("water.hessian.info");
    hessianSettings.set_displacementByUser(0.002 * angstrom);

    EXPECT_EQ(hessianSettings.get_hessianFile(), "water.hessian");
    EXPECT_EQ(hessianSettings.get_hessianInfoFile(), "water.hessian.info");
    EXPECT_EQ(hessianSettings.get_displacement(), 0.002 * angstrom);
}

TEST(TestHessianSettings, setOptimizeBeforeHessian)
{
    HessianSettings hessianSettings;
    hessianSettings.set_optimizeBeforeHessianByUser(false);
    EXPECT_FALSE(hessianSettings.get_optimizeBeforeHessian());

    hessianSettings.set_optimizeBeforeHessianByUser(true);
    EXPECT_TRUE(hessianSettings.get_optimizeBeforeHessian());
}
