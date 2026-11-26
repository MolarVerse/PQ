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
#include "referenceSettings.hpp"
#include "units.hpp"

TEST(ManostatSettingsTest, DoubleSettersAndGetters)
{
    ManostatSettings settings;
    settings.set_targetPressureByUser(2.5 * bar);
    EXPECT_EQ(settings.get_targetPressure(), 2.5 * bar);

    settings.set_tauByUser(1.5 * ps);
    EXPECT_EQ(settings.get_tau(), 1.5 * ps);

    settings.set_compressibilityByUser(4.5e-5 / bar);
    EXPECT_EQ(settings.get_compressibility(), 4.5e-5 / bar);
}

TEST(ManostatSettingsTest, SetFixedAxisViaEnum)
{
    ManostatSettings settings;

    settings.set_fixedAxisByUser(FixedAxis::XY);
    EXPECT_EQ(settings.get_fixedAxis(), FixedAxis::XY);
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
    settings.set_manostatTypeByUser(ManostatType::NONE);
    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings);
    EXPECT_EQ(settings.get_fixedAxis(), FixedAxis::ALL);

    // When manostat is selected, default becomes NONE
    ManostatSettings settings2;
    settings2.set_manostatTypeByUser(ManostatType::BERENDSEN);
    settings2.finalize(referenceSettings);
    EXPECT_EQ(settings2.get_fixedAxis(), FixedAxis::NONE);

    ManostatSettings settings3;
    settings3.set_manostatTypeByUser(ManostatType::STOCHASTIC_RESCALING);
    settings3.finalize(referenceSettings);
    EXPECT_EQ(settings3.get_fixedAxis(), FixedAxis::NONE);

    // When explicitly set, it overrides the default and remains set
    ManostatSettings settings4;
    settings4.set_manostatTypeByUser(ManostatType::BERENDSEN);
    settings4.set_fixedAxisByUser(FixedAxis::Z);
    settings4.finalize(referenceSettings);
    EXPECT_EQ(settings4.get_fixedAxis(), FixedAxis::Z);

    // Changing manostat type does not override explicitly set fixed axis
    settings4.set_manostatTypeByUser(ManostatType::BERENDSEN);
    settings4.finalize(referenceSettings);
    EXPECT_EQ(settings4.get_fixedAxis(), FixedAxis::Z);
}

/**
 * @brief finalize() must add the reference file matching the configured
 * manostat type.
 */
TEST(ManostatSettingsTest, FinalizeAddsReferenceFileForManostatType)
{
    ManostatSettings berendsen;
    berendsen.set_manostatTypeByUser(ManostatType::BERENDSEN);
    ReferenceSettings berendsenReferences;
    berendsen.finalize(berendsenReferences);
    EXPECT_TRUE(berendsenReferences.getReferenceFileNames().contains(
        ReferenceFiles::BERENDSEN_FILE
    ));

    ManostatSettings stochasticRescaling;
    stochasticRescaling.set_manostatTypeByUser(
        ManostatType::STOCHASTIC_RESCALING
    );
    ReferenceSettings stochasticReferences;
    stochasticRescaling.finalize(stochasticReferences);
    EXPECT_TRUE(stochasticReferences.getReferenceFileNames().contains(
        ReferenceFiles::STOCHASTIC_RESCALING_FILE
    ));
}

/**
 * @brief finalize() must not add any reference file when no manostat type
 * has been set.
 */
TEST(ManostatSettingsTest, FinalizeAddsNoReferenceFileWhenManostatTypeUnset)
{
    ManostatSettings settings;

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings);

    EXPECT_TRUE(referenceSettings.getReferenceFileNames().empty());
}

/**
 * @brief finalize() must not add any reference file when the manostat
 * type is explicitly NONE.
 */
TEST(ManostatSettingsTest, FinalizeAddsNoReferenceFileForManostatTypeNone)
{
    ManostatSettings settings;
    settings.set_manostatTypeByUser(ManostatType::NONE);

    ReferenceSettings referenceSettings;
    settings.finalize(referenceSettings);

    EXPECT_TRUE(referenceSettings.getReferenceFileNames().empty());
}
