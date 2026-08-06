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

#include "fileSettings.hpp"
#include "gtest/gtest.h"

TEST(FileSettingsTest, MolDescriptorRoundTrip)
{
    settings::FileSettings::setMolDescriptorFileName("mol.dat");
    EXPECT_EQ(settings::FileSettings::getMolDescriptorFileName(), "mol.dat");
}

TEST(FileSettingsTest, GuffDatRoundTrip)
{
    settings::FileSettings::setGuffDatFileName("guff.dat");
    EXPECT_EQ(settings::FileSettings::getGuffDatFileName(), "guff.dat");
}

TEST(FileSettingsTest, TopologyAndParameterRoundTrip)
{
    settings::FileSettings::setTopologyFileName("topology.top");
    EXPECT_EQ(settings::FileSettings::getTopologyFileName(), "topology.top");

    settings::FileSettings::setParameterFileName("parameter.par");
    EXPECT_EQ(settings::FileSettings::getParameterFilename(), "parameter.par");
}

TEST(FileSettingsTest, StartAndRingPolymerRoundTrip)
{
    settings::FileSettings::setStartFileName("start.rst");
    EXPECT_EQ(settings::FileSettings::getStartFileName(), "start.rst");

    settings::FileSettings::setRingPolymerStartFileName("rp_start.rst");
    EXPECT_EQ(
        settings::FileSettings::getRingPolymerStartFileName(),
        "rp_start.rst"
    );
}

TEST(FileSettingsTest, MShakeAndDFTBRoundTrip)
{
    settings::FileSettings::setMShakeFileName("mshake.dat");
    EXPECT_EQ(settings::FileSettings::getMShakeFileName(), "mshake.dat");

    settings::FileSettings::setDFTBFileName("dftb_in.hsd");
    EXPECT_EQ(settings::FileSettings::getDFTBFileName(), "dftb_in.hsd");
}

TEST(FileSettingsTest, IntraNonBondedRoundTrip)
{
    settings::FileSettings::setIntraNonBondedFileName("intra.dat");
    EXPECT_EQ(
        settings::FileSettings::getIntraNonBondedFileName(),
        "intra.dat"
    );
}
