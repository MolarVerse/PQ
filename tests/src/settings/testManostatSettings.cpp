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

#include "gtest/gtest.h"
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
    EXPECT_DOUBLE_EQ(
        settings::ManostatSettings::getCompressibility(),
        4.5e-5
    );
}

TEST(ManostatSettingsTest, AnisotropicAxesSettersAndGetters)
{
    settings::ManostatSettings::set2DIsotropicAxes({0u, 1u});
    EXPECT_EQ(
        settings::ManostatSettings::get2DIsotropicAxes(),
        (std::vector<size_t>{0u, 1u})
    );

    settings::ManostatSettings::set2DAnisotropicAxis(2u);
    EXPECT_EQ(settings::ManostatSettings::get2DAnisotropicAxis(), 2u);
}

TEST(ManostatSettingsTest, StringRoundTripForManostatType)
{
    EXPECT_EQ(
        settings::string(settings::ManostatType::BERENDSEN),
        "berendsen"
    );
    EXPECT_EQ(
        settings::string(settings::ManostatType::STOCHASTIC_RESCALING),
        "stochastic_rescaling"
    );
    EXPECT_EQ(settings::string(settings::ManostatType::NONE), "none");
}

TEST(ManostatSettingsTest, StringRoundTripForIsotropy)
{
    EXPECT_EQ(
        settings::string(settings::Isotropy::ISOTROPIC),
        "isotropic"
    );
    EXPECT_EQ(
        settings::string(settings::Isotropy::SEMI_ISOTROPIC),
        "semi_isotropic"
    );
    EXPECT_EQ(
        settings::string(settings::Isotropy::ANISOTROPIC),
        "anisotropic"
    );
    EXPECT_EQ(
        settings::string(settings::Isotropy::FULL_ANISOTROPIC),
        "full_anisotropic"
    );
}
