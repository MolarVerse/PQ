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
    settings.set_molDescriptorFileByUser("mol.dat");
    EXPECT_EQ(settings.get_molDescriptorFile(), "mol.dat");
}

TEST(FileSettingsTest, GuffDatRoundTrip)
{
    FileSettings settings;
    settings.set_guffDatFileByUser("guff.dat");
    EXPECT_EQ(settings.get_guffDatFile(), "guff.dat");
}

TEST(FileSettingsTest, TopologyAndParameterRoundTrip)
{
    FileSettings settings;
    settings.set_topologyFileByUser("topology.top");
    EXPECT_EQ(settings.get_topologyFile(), "topology.top");

    settings.set_parameterFileByUser("parameter.par");
    EXPECT_EQ(settings.get_parameterFile(), "parameter.par");
}

TEST(FileSettingsTest, StartAndRingPolymerRoundTrip)
{
    FileSettings settings;
    settings.set_startFileByUser("start.rst");
    EXPECT_EQ(settings.get_startFile(), "start.rst");

    settings.set_rpmdStartFileByUser("rp_start.rst");
    EXPECT_EQ(settings.get_rpmdStartFile(), "rp_start.rst");
}

TEST(FileSettingsTest, MShakeAndDFTBRoundTrip)
{
    FileSettings settings;
    settings.set_mShakeFileByUser("mshake.dat");
    EXPECT_EQ(settings.get_mShakeFile(), "mshake.dat");

    settings.set_dftbFileByUser("dftb_in.hsd");
    EXPECT_EQ(settings.get_dftbFile(), "dftb_in.hsd");

    settings.set_tmFileByUser("tm_in.hsd");
    EXPECT_EQ(settings.get_tmFile(), "tm_in.hsd");
}

TEST(FileSettingsTest, IntraNonBondedRoundTrip)
{
    FileSettings settings;
    settings.set_intraNonBondedFileByUser("intra.dat");
    EXPECT_EQ(settings.get_intraNonBondedFile(), "intra.dat");
}
