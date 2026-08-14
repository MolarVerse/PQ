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

#include "externalQMScripts.hpp"

using settings::QMMethod;

TEST(ExternalQMScriptsTest, catalogsSupportedPrograms)
{
    ASSERT_EQ(cli::externalQMMethods.size(), 3);
    EXPECT_EQ(
        cli::externalQMProgramName(cli::externalQMMethods[0]),
        "dftbplus"
    );
    EXPECT_EQ(cli::externalQMProgramName(cli::externalQMMethods[1]), "pyscf");
    EXPECT_EQ(
        cli::externalQMProgramName(cli::externalQMMethods[2]),
        "turbomole"
    );
}

TEST(ExternalQMScriptsTest, describesBundledScripts)
{
    const auto dftbPlus = cli::externalQMScripts(QMMethod::DFTBPLUS);
    ASSERT_EQ(dftbPlus.size(), 1);
    EXPECT_EQ(dftbPlus.front().name, "dftbplus_periodic_stress");
    EXPECT_EQ(dftbPlus.front().requiredFileKeyword, "dftb_file");
    EXPECT_EQ(
        cli::recommendedExternalQMScript(QMMethod::DFTBPLUS),
        dftbPlus.front().name
    );

    const auto pyScf = cli::externalQMScripts(QMMethod::PYSCF);
    ASSERT_EQ(pyScf.size(), 2);
    EXPECT_EQ(pyScf.front().name, "pyscf_hf.py");
    EXPECT_EQ(pyScf.back().name, "pyscf_mp2.py");
    EXPECT_TRUE(cli::recommendedExternalQMScript(QMMethod::PYSCF).empty());

    const auto turbomole = cli::externalQMScripts(QMMethod::TURBOMOLE);
    ASSERT_EQ(turbomole.size(), 1);
    EXPECT_EQ(turbomole.front().name, "turbomole_ricc2");
    EXPECT_EQ(turbomole.front().requiredWorkingFile, "tm_define.template");
}

TEST(ExternalQMScriptsTest, identifiesCatalogEntries)
{
    EXPECT_TRUE(
        cli::isExternalQMScript(QMMethod::DFTBPLUS, "dftbplus_periodic_stress")
    );
    EXPECT_FALSE(cli::isExternalQMScript(QMMethod::DFTBPLUS, "pyscf_hf.py"));
    EXPECT_TRUE(cli::externalQMScripts(QMMethod::MACE).empty());
    EXPECT_TRUE(cli::externalQMProgramName(QMMethod::MACE).empty());
}
