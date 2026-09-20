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

#include "manostatSettings.hpp"

TEST(ManostatSettingsTest, SetManostatTypeViaString)
{
    settings::ManostatSettings::setManostatType("berendsen");
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        settings::ManostatType::BERENDSEN
    );

    settings::ManostatSettings::setManostatType("stochastic_rescaling");
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        settings::ManostatType::STOCHASTIC_RESCALING
    );

    settings::ManostatSettings::setManostatType("none");
    EXPECT_EQ(
        settings::ManostatSettings::getManostatType(),
        settings::ManostatType::NONE
    );
}

TEST(ManostatSettingsTest, SetIsotropyViaString)
{
    settings::ManostatSettings::setIsotropy("isotropic");
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::ISOTROPIC
    );

    settings::ManostatSettings::setIsotropy("semi_isotropic");
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::SEMI_ISOTROPIC
    );

    settings::ManostatSettings::setIsotropy("anisotropic");
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::ANISOTROPIC
    );

    settings::ManostatSettings::setIsotropy("full_anisotropic");
    EXPECT_EQ(
        settings::ManostatSettings::getIsotropy(),
        settings::Isotropy::FULL_ANISOTROPIC
    );
}

TEST(ManostatSettingsTest, DoubleSettersAndGetters)
{
    settings::ManostatSettings::setTargetPressure(2.5);
    EXPECT_DOUBLE_EQ(settings::ManostatSettings::getTargetPressure(), 2.5);

    settings::ManostatSettings::setTauManostat(1.5);
    EXPECT_DOUBLE_EQ(settings::ManostatSettings::getTauManostat(), 1.5);

    settings::ManostatSettings::setCompressibility(4.5e-5);
    EXPECT_DOUBLE_EQ(settings::ManostatSettings::getCompressibility(), 4.5e-5);
}

TEST(ManostatSettingsTest, AnisotropicAxesSettersAndGetters)
{
    settings::ManostatSettings::set2DIsotropicAxes({0U, 1U});
    EXPECT_EQ(
        settings::ManostatSettings::get2DIsotropicAxes(),
        (std::vector<size_t>{0U, 1U})
    );

    settings::ManostatSettings::set2DAnisotropicAxis(2U);
    EXPECT_EQ(settings::ManostatSettings::get2DAnisotropicAxis(), 2U);
}

TEST(ManostatSettingsTest, StringRoundTripForManostatType)
{
    EXPECT_EQ(settings::string(settings::ManostatType::BERENDSEN), "berendsen");
    EXPECT_EQ(
        settings::string(settings::ManostatType::STOCHASTIC_RESCALING),
        "stochastic_rescaling"
    );
    EXPECT_EQ(settings::string(settings::ManostatType::NONE), "none");
}

TEST(ManostatSettingsTest, StringRoundTripForIsotropy)
{
    EXPECT_EQ(settings::string(settings::Isotropy::ISOTROPIC), "isotropic");
    EXPECT_EQ(
        settings::string(settings::Isotropy::SEMI_ISOTROPIC),
        "semi_isotropic"
    );
    EXPECT_EQ(settings::string(settings::Isotropy::ANISOTROPIC), "anisotropic");
    EXPECT_EQ(
        settings::string(settings::Isotropy::FULL_ANISOTROPIC),
        "full_anisotropic"
    );
    EXPECT_EQ(settings::string(settings::Isotropy::NONE), "isotropic");
}

TEST(ManostatSettingsTest, SetFixedAxisViaEnum)
{
    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::XY);
    EXPECT_EQ(
        settings::ManostatSettings::getFixedAxis(),
        settings::FixedAxis::XY
    );
}

TEST(ManostatSettingsTest, StringRoundTripForFixedAxis)
{
    EXPECT_EQ(settings::string(settings::FixedAxis::NONE), "none");
    EXPECT_EQ(settings::string(settings::FixedAxis::X), "x");
    EXPECT_EQ(settings::string(settings::FixedAxis::Y), "y");
    EXPECT_EQ(settings::string(settings::FixedAxis::Z), "z");
    EXPECT_EQ(settings::string(settings::FixedAxis::XY), "xy");
    EXPECT_EQ(settings::string(settings::FixedAxis::XZ), "xz");
    EXPECT_EQ(settings::string(settings::FixedAxis::YZ), "yz");
    EXPECT_EQ(settings::string(settings::FixedAxis::ALL), "all");
}

TEST(ManostatSettingsTest, FixedAxisBitwiseOperators)
{
    using enum settings::FixedAxis;

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
    using enum settings::FixedAxis;

    EXPECT_TRUE(settings::isAxisFixed(X, 0));
    EXPECT_FALSE(settings::isAxisFixed(X, 1));
    EXPECT_FALSE(settings::isAxisFixed(X, 2));

    EXPECT_TRUE(settings::isAxisFixed(XY, 0));
    EXPECT_TRUE(settings::isAxisFixed(XY, 1));
    EXPECT_FALSE(settings::isAxisFixed(XY, 2));

    EXPECT_TRUE(settings::isAxisFixed(ALL, 0));
    EXPECT_TRUE(settings::isAxisFixed(ALL, 1));
    EXPECT_TRUE(settings::isAxisFixed(ALL, 2));

    EXPECT_FALSE(settings::isAxisFixed(NONE, 0));
    EXPECT_FALSE(settings::isAxisFixed(NONE, 1));
    EXPECT_FALSE(settings::isAxisFixed(NONE, 2));
}

TEST(ManostatSettingsTest, FixedAxisDefaultBehavior)
{
    // When no manostat is selected and fixed axis was not set, default is ALL
    settings::ManostatSettings::setIsFixedAxisSet(false);
    settings::ManostatSettings::setManostatType(settings::ManostatType::NONE);
    EXPECT_EQ(
        settings::ManostatSettings::getFixedAxis(),
        settings::FixedAxis::ALL
    );
    EXPECT_FALSE(settings::ManostatSettings::isFixedAxisSet());

    // When manostat is selected, default becomes NONE
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::BERENDSEN
    );
    EXPECT_EQ(
        settings::ManostatSettings::getFixedAxis(),
        settings::FixedAxis::NONE
    );

    settings::ManostatSettings::setManostatType(
        settings::ManostatType::STOCHASTIC_RESCALING
    );
    EXPECT_EQ(
        settings::ManostatSettings::getFixedAxis(),
        settings::FixedAxis::NONE
    );

    // When explicitly set, it overrides the default and remains set
    settings::ManostatSettings::setFixedAxis(settings::FixedAxis::Z);
    EXPECT_TRUE(settings::ManostatSettings::isFixedAxisSet());
    EXPECT_EQ(
        settings::ManostatSettings::getFixedAxis(),
        settings::FixedAxis::Z
    );

    // Changing manostat type does not override explicitly set fixed axis
    settings::ManostatSettings::setManostatType(
        settings::ManostatType::BERENDSEN
    );
    EXPECT_EQ(
        settings::ManostatSettings::getFixedAxis(),
        settings::FixedAxis::Z
    );

    // Reset back to unset for other tests
    settings::ManostatSettings::setIsFixedAxisSet(false);
    settings::ManostatSettings::setManostatType(settings::ManostatType::NONE);
}
