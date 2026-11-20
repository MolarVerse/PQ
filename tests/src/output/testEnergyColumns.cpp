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

#include <string_view>
#include <vector>

#include "constraintSettings.hpp"
#include "energyColumns.hpp"
#include "forceFieldSettings.hpp"
#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "physicalData.hpp"
#include "settings.hpp"
#include "testEnergyOutput.hpp"
#include "thermostatSettings.hpp"

/**
 * @brief tests the order of all columns with every option active
 *
 */
TEST_F(TestEnergyOutput, energyColumnsAllActive)
{
    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::ON);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::Z);
    _settings->thermostatSettings.setThermostatTypeByUser(
        ThermostatType::NOSE_HOOVER
    );
    _settings->constraintSettings.activateDistanceConstraintsByUser();
    settings::GeneralSettings::setJobtype(JobType::QMMM_MD);

    const out::EnergyColumns columns(*_physicalData, *_settings);

    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    std::vector<std::string_view> names;
    for (const auto &column : columns.get())
    {
        names.push_back(column.name);
        EXPECT_EQ(column.aloneInInfoRow, column.name == "P(COUPLED)");
    }

    const std::vector<std::string_view> expected{
        "TEMPERATURE",
        "PRESSURE",
        "E(TOT)",
        "E(QM)",
        "N(QM-ATOMS)",
        "E(KIN)",
        "E(INTRA)",
        "E(COUL)",
        "E(NON-COUL)",
        "E(BOND)",
        "E(ANGLE)",
        "E(DIHEDRAL)",
        "E(IMPROPER)",
        "N(SM-MOL)",
        "E(QM-REF)",
        "VOLUME",
        "DENSITY",
        "P(COUPLED)",
        "E(NH-MOMENTUM)",
        "E(NH-FRICTION)",
        "LOWER-DIST-CONSTR",
        "UPPER-DIST-CONSTR",
        "MOMENTUM",
        "LOOPTIME"
    };

    EXPECT_EQ(names, expected);
}

/**
 * @brief tests the minimal column set of a QM job
 *
 */
TEST_F(TestEnergyOutput, energyColumnsQM)
{
    _physicalData->setQMEnergy(7.0);
    _physicalData->setQMReferenceEnergy(2.0);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::NONE);
    settings::GeneralSettings::setJobtype(JobType::QM_MD);

    const out::EnergyColumns columns(*_physicalData, *_settings);

    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    const auto active = columns.get();

    ASSERT_EQ(active.size(), 9);
    EXPECT_EQ(active[3].name, "E(QM)");
    EXPECT_DOUBLE_EQ(active[3].value, 5.0);
    EXPECT_EQ(active[7].format, out::EnergyColumnFormat::SCIENTIFIC);
    EXPECT_EQ(active[8].format, out::EnergyColumnFormat::TIME);
}

/**
 * @brief tests column labels and the header lines
 *
 */
TEST_F(TestEnergyOutput, energyColumnLabelAndHeader)
{
    EXPECT_EQ(
        out::energyColumnLabel({.name = "E(TOT)", .unit = "kcal/mol"}),
        "E(TOT)[kcal/mol]"
    );
    EXPECT_EQ(
        out::energyColumnLabel({.name = "N(QM-ATOMS)", .unit = "-"}),
        "N(QM-ATOMS)[-]"
    );

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::NONE);
    settings::GeneralSettings::setJobtype(JobType::QM_MD);

    const out::EnergyColumns columns(*_physicalData, *_settings);

    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    EXPECT_EQ(
        out::energyFileHeader(columns),
        "#STEP[-] TEMPERATURE[K] PRESSURE[bar] E(TOT)[kcal/mol] "
        "E(QM)[kcal/mol] N(QM-ATOMS)[-] E(KIN)[kcal/mol] E(INTRA)[kcal/mol] "
        "MOMENTUM[amuA/fs] LOOPTIME[s]"
    );
    EXPECT_TRUE(out::energyFileMarker().starts_with("# PQ energy file "));
}
