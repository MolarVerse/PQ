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

#include "enums/manostat.hpp"
#include "manostatSettings.hpp"

TEST(ManostatSettingsTest, DoubleSettersAndGetters)
{
    ManostatSettings settings;
    settings.setTargetPressure(2.5);
    EXPECT_DOUBLE_EQ(settings.getTargetPressure(), 2.5);

    settings.setTauManostat(1.5);
    EXPECT_DOUBLE_EQ(settings.getTauManostat(), 1.5);

    settings.setCompressibility(4.5e-5);
    EXPECT_DOUBLE_EQ(settings.getCompressibility(), 4.5e-5);
}

TEST(ManostatSettingsTest, SetFixedAxisViaEnum)
{
    ManostatSettings settings;

    settings.setFixedAxis(FixedAxis::XY);
    EXPECT_EQ(settings.getFixedAxis(), FixedAxis::XY);
}

TEST(ManostatSettingsTest, FixedAxisBitwiseOperators)
{
    using enum FixedAxis;

    EXPECT_EQ(X | Y, XY);
    EXPECT_EQ(X | Z, XZ);
    EXPECT_EQ(Y | Z, YZ);
    EXPECT_EQ(XY | Z, ALL);
    EXPECT_EQ(X | Y | Z, ALL);

    EXPECT_EQ((XY & X), X);
    EXPECT_EQ((XY & Z), NONE);
    EXPECT_EQ((ALL & XZ), XZ);

    auto axis  = X;
    axis      |= Y;
    EXPECT_EQ(axis, XY);

    axis &= X;
    EXPECT_EQ(axis, X);

    EXPECT_EQ(~NONE & ALL, ALL);
    EXPECT_EQ(~ALL & ALL, NONE);
    EXPECT_EQ(~X & ALL, YZ);
    EXPECT_EQ(~XY & ALL, Z);
}

TEST(ManostatSettingsTest, FixedAxisHelperFunctions)
{
    using enum FixedAxis;

    EXPECT_TRUE(isAxisFixed(X, 0));
    EXPECT_FALSE(isAxisFixed(X, 1));
    EXPECT_FALSE(isAxisFixed(X, 2));

    EXPECT_TRUE(isAxisFixed(XY, 0));
    EXPECT_TRUE(isAxisFixed(XY, 1));
    EXPECT_FALSE(isAxisFixed(XY, 2));

    EXPECT_TRUE(isAxisFixed(ALL, 0));
    EXPECT_TRUE(isAxisFixed(ALL, 1));
    EXPECT_TRUE(isAxisFixed(ALL, 2));

    EXPECT_FALSE(isAxisFixed(NONE, 0));
    EXPECT_FALSE(isAxisFixed(NONE, 1));
    EXPECT_FALSE(isAxisFixed(NONE, 2));
}

TEST(ManostatSettingsTest, FixedAxisDefaultBehavior)
{
    // When no manostat is selected and fixed axis was not set, default is ALL
    ManostatSettings settings;
    settings.setManostatType(ManostatType::NONE);
    EXPECT_FALSE(settings.isFixedAxisSet());
    settings.finalize();
    EXPECT_EQ(settings.getFixedAxis(), FixedAxis::ALL);

    // When manostat is selected, default becomes NONE
    ManostatSettings settings2;
    settings2.setManostatType(ManostatType::BERENDSEN);
    settings2.finalize();
    EXPECT_EQ(settings2.getFixedAxis(), FixedAxis::NONE);

    ManostatSettings settings3;
    settings3.setManostatType(ManostatType::STOCHASTIC_RESCALING);
    settings3.finalize();
    EXPECT_EQ(settings3.getFixedAxis(), FixedAxis::NONE);

    // When explicitly set, it overrides the default and remains set
    ManostatSettings settings4;
    settings4.setManostatType(ManostatType::BERENDSEN);
    settings4.setFixedAxis(FixedAxis::Z);
    settings4.finalize();
    EXPECT_TRUE(settings4.isFixedAxisSet());
    EXPECT_EQ(settings4.getFixedAxis(), FixedAxis::Z);

    // Changing manostat type does not override explicitly set fixed axis
    settings4.setManostatType(ManostatType::BERENDSEN);
    settings4.finalize();
    EXPECT_EQ(settings4.getFixedAxis(), FixedAxis::Z);
}
