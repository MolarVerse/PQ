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

TEST(FileSettingsTest, MolDescriptorRoundTrip)
{
    FileSettings settings;
    settings.setMolDescriptorFileNameByUser("mol.dat");
    EXPECT_EQ(settings.getMolDescriptorFileName(), "mol.dat");
}

TEST(FileSettingsTest, GuffDatRoundTrip)
{
    FileSettings settings;
    settings.setGuffDatFileNameByUser("guff.dat");
    EXPECT_EQ(settings.getGuffDatFileName(), "guff.dat");
}

TEST(FileSettingsTest, TopologyAndParameterRoundTrip)
{
    FileSettings::setTopologyFileName("topology.top");
    EXPECT_EQ(FileSettings::getTopologyFileName(), "topology.top");

    FileSettings::setParameterFileName("parameter.par");
    EXPECT_EQ(FileSettings::getParameterFilename(), "parameter.par");
}

TEST(FileSettingsTest, StartAndRingPolymerRoundTrip)
{
    FileSettings::setStartFileName("start.rst");
    EXPECT_EQ(FileSettings::getStartFileName(), "start.rst");

    FileSettings::setRingPolymerStartFileName("rp_start.rst");
    EXPECT_EQ(FileSettings::getRingPolymerStartFileName(), "rp_start.rst");
}

TEST(FileSettingsTest, MShakeAndDFTBRoundTrip)
{
    FileSettings::setMShakeFileName("mshake.dat");
    EXPECT_EQ(FileSettings::getMShakeFileName(), "mshake.dat");

    FileSettings::setDFTBFileName("dftb_in.hsd");
    EXPECT_EQ(FileSettings::getDFTBFileName(), "dftb_in.hsd");
}

TEST(FileSettingsTest, IntraNonBondedRoundTrip)
{
    FileSettings::setIntraNonBondedFileName("intra.dat");
    EXPECT_EQ(FileSettings::getIntraNonBondedFileName(), "intra.dat");
}
