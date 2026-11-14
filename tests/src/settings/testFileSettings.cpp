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
    FileSettings settings;
    settings.setTopologyFileNameByUser("topology.top");
    EXPECT_EQ(settings.getTopologyFileName(), "topology.top");

    settings.setParameterFileNameByUser("parameter.par");
    EXPECT_EQ(settings.getParameterFileName(), "parameter.par");
}

TEST(FileSettingsTest, StartAndRingPolymerRoundTrip)
{
    FileSettings settings;
    settings.setStartFileNameByUser("start.rst");
    EXPECT_EQ(settings.getStartFileName(), "start.rst");

    settings.setRingPolymerStartFileNameByUser("rp_start.rst");
    EXPECT_EQ(settings.getRingPolymerStartFileName(), "rp_start.rst");
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
    FileSettings settings;
    settings.setIntraNonBondedFileNameByUser("intra.dat");
    EXPECT_EQ(settings.getIntraNonBondedFileName(), "intra.dat");
}
