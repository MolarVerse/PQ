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

#include <iosfwd>
#include <iterator>
#include <string>

#include "enums/potential.hpp"
#include "forceFieldSettings.hpp"
#include "generalSettings.hpp"
#include "infoOutput.hpp"
#include "manostatSettings.hpp"
#include "physicalData.hpp"
#include "settings.hpp"
#include "testEnergyOutput.hpp"
#include "thermostatSettings.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "vector3d.hpp"

/**
 * @brief tests writing info file
 *
 * @details basic info file
 *
 */
TEST_F(TestEnergyOutput, writeInfoForceFieldNotActive)
{
    _physicalData->set_temperature(mstd::units::point<K>(1.0));
    _physicalData->set_pressure(2.0 * bar);
    _physicalData->set_kineticEnergy(3.0 * kcal_per_mol);
    _physicalData->set_coulombEnergy(4.0 * kcal_per_mol);
    _physicalData->set_nonCoulombEnergy(5.0 * kcal_per_mol);
    _physicalData->set_momentum(linalg::Vec3D(6.0) * g_per_mol * angstrom / fs);
    _physicalData->set_intraCoulombEnergy(9.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(10.0 * kcal_per_mol);
    _physicalData->set_loopTime(0.1 * s);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    _infoOutput->setFilename("default.info", false);
    _infoOutput->write(100.0 * ps, *_physicalData, *_settings);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|                                     PQ info file                    "
        "                  |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   SIMULATION-TIME       100.00000 ps       TEMPERATURE             "
        "1.00000 K          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   PRESSURE                2.00000 bar      E(TOT)                 "
        "12.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               "
        "19.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             "
        "5.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                "
        "0.10000 s          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
}

/**
 * @brief tests writing info file
 *
 * @details force field is active
 *
 */
TEST_F(TestEnergyOutput, writeInfoForceFieldActive)
{
    _physicalData->set_temperature(mstd::units::point<K>(1.0));
    _physicalData->set_pressure(2.0 * bar);
    _physicalData->set_kineticEnergy(3.0 * kcal_per_mol);
    _physicalData->set_coulombEnergy(4.0 * kcal_per_mol);
    _physicalData->set_nonCoulombEnergy(5.0 * kcal_per_mol);
    _physicalData->set_momentum(linalg::Vec3D(6.0) * g_per_mol * angstrom / fs);
    _physicalData->set_intraCoulombEnergy(9.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(10.0 * kcal_per_mol);

    _physicalData->set_bondEnergy(19.0 * kcal_per_mol);
    _physicalData->set_angleEnergy(20.0 * kcal_per_mol);
    _physicalData->set_dihedralEnergy(21.0 * kcal_per_mol);
    _physicalData->set_improperEnergy(22.0 * kcal_per_mol);
    _physicalData->set_loopTime(0.1 * s);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::ON);
    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    _infoOutput->setFilename("default.info", false);
    _infoOutput->write(100.0 * ps, *_physicalData, *_settings);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|                                     PQ info file                    "
        "                  |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   SIMULATION-TIME       100.00000 ps       TEMPERATURE             "
        "1.00000 K          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   PRESSURE                2.00000 bar      E(TOT)                 "
        "94.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               "
        "19.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             "
        "5.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(BOND)                19.00000 kcal/mol E(ANGLE)               "
        "20.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(DIHEDRAL)            21.00000 kcal/mol E(IMPROPER)            "
        "22.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                "
        "0.10000 s          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
}

/**
 * @brief tests writing info file
 *
 * @details manostat is active
 *
 */
TEST_F(TestEnergyOutput, writeInfoManostatIsActive)
{
    _physicalData->set_temperature(toTemperature(1.0));
    _physicalData->set_pressure(2.0 * bar);
    _physicalData->set_kineticEnergy(3.0 * kcal_per_mol);
    _physicalData->set_coulombEnergy(4.0 * kcal_per_mol);
    _physicalData->set_nonCoulombEnergy(5.0 * kcal_per_mol);
    _physicalData->set_momentum(linalg::Vec3D(6.0) * g_per_mol * angstrom / fs);
    _physicalData->set_intraCoulombEnergy(9.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(10.0 * kcal_per_mol);

    _physicalData->set_volume(11.0 * angstrom3);
    _physicalData->set_density(12.0 * kg_per_litre);
    _physicalData->set_loopTime(0.1 * s);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::NONE);
    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    _infoOutput->setFilename("default.info", false);
    _infoOutput->write(100.0 * ps, *_physicalData, *_settings);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|                                     PQ info file                    "
        "                  |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   SIMULATION-TIME       100.00000 ps       TEMPERATURE             "
        "1.00000 K          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   PRESSURE                2.00000 bar      E(TOT)                 "
        "12.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               "
        "19.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             "
        "5.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   VOLUME                 11.00000 A^3      DENSITY                "
        "12.00000 g/cm^3     |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                "
        "0.10000 s          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
}

/**
 * @brief tests writing info file with fixed axis
 *
 * @details manostat and fixed_axis are active
 *
 */
TEST_F(TestEnergyOutput, writeInfoManostatIsActiveWithFixedAxis)
{
    _physicalData->set_temperature(toTemperature(1.0));
    _physicalData->set_pressure(2.0 * bar);
    _physicalData->set_coupledPressure(2.5 * bar);
    _physicalData->set_kineticEnergy(3.0 * kcal_per_mol);
    _physicalData->set_coulombEnergy(4.0 * kcal_per_mol);
    _physicalData->set_nonCoulombEnergy(5.0 * kcal_per_mol);
    _physicalData->set_momentum(linalg::Vec3D(6.0) * g_per_mol * angstrom / fs);
    _physicalData->set_intraCoulombEnergy(9.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(10.0 * kcal_per_mol);

    _physicalData->set_volume(11.0 * angstrom3);
    _physicalData->set_density(12.0 * kg_per_litre);
    _physicalData->set_loopTime(0.1 * s);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::BERENDSEN);
    _settings->manostatSettings.setFixedAxisByUser(FixedAxis::Z);
    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    _infoOutput->setFilename("default.info", false);
    _infoOutput->write(100.0 * ps, *_physicalData, *_settings);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|                                     PQ info file                    "
        "                  |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   SIMULATION-TIME       100.00000 ps       TEMPERATURE             "
        "1.00000 K          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   PRESSURE                2.00000 bar      E(TOT)                 "
        "12.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               "
        "19.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             "
        "5.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   VOLUME                 11.00000 A^3      DENSITY                "
        "12.00000 g/cm^3     |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   P(COUPLED)              2.50000 bar                               "
        "                  |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                "
        "0.10000 s          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
}

/**
 * @brief tests writing info file
 *
 * @details qm is active
 *
 */
TEST_F(TestEnergyOutput, writeInfoQmIsActive)
{
    _physicalData->reset();

    _physicalData->set_temperature(toTemperature(1.0));
    _physicalData->set_pressure(2.0 * bar);
    _physicalData->set_kineticEnergy(3.0 * kcal_per_mol);
    _physicalData->set_momentum(linalg::Vec3D(6.0) * g_per_mol * angstrom / fs);
    _physicalData->set_intraCoulombEnergy(0.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(0.0 * kcal_per_mol);

    _physicalData->set_qmEnergy(5.0 * kcal_per_mol);

    _physicalData->set_volume(19.0 * angstrom3);
    _physicalData->set_density(20.0 * kg_per_litre);
    _physicalData->set_loopTime(0.1 * s);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    settings::GeneralSettings::setJobtype(JobType::QM_MD);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::NONE);

    _infoOutput->setFilename("default.info", false);
    _infoOutput->write(100.0 * ps, *_physicalData, *_settings);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|                                     PQ info file                    "
        "                  |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   SIMULATION-TIME       100.00000 ps       TEMPERATURE             "
        "1.00000 K          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   PRESSURE                2.00000 bar      E(TOT)                  "
        "8.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(QM)                   5.00000 kcal/mol N(QM-ATOMS)             "
        "0.00000 -          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(KIN)                  3.00000 kcal/mol E(INTRA)                "
        "0.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                "
        "0.10000 s          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
}

/**
 * @brief tests writing info file
 *
 * @details nose hoover is active
 *
 */
TEST_F(TestEnergyOutput, writeInfoNoseHooverActive)
{
    _physicalData->set_temperature(toTemperature(1.0));
    _physicalData->set_pressure(2.0 * bar);
    _physicalData->set_kineticEnergy(3.0 * kcal_per_mol);
    _physicalData->set_coulombEnergy(4.0 * kcal_per_mol);
    _physicalData->set_nonCoulombEnergy(5.0 * kcal_per_mol);
    _physicalData->set_momentum(linalg::Vec3D(6.0) * g_per_mol * angstrom / fs);
    _physicalData->set_intraCoulombEnergy(9.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(10.0 * kcal_per_mol);
    _physicalData->set_noseHooverMomentumEnergy(11.0 * kcal_per_mol);
    _physicalData->set_noseHooverFrictionEnergy(12.0);
    _physicalData->set_loopTime(0.1 * s);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    settings::GeneralSettings::setJobtype(JobType::MM_MD);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::NONE);
    _settings->thermostatSettings.setThermostatTypeByUser(
        ThermostatType::NOSE_HOOVER
    );

    _infoOutput->setFilename("default.info", false);
    _infoOutput->write(100.0 * ps, *_physicalData, *_settings);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|                                     PQ info file                    "
        "                  |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   SIMULATION-TIME       100.00000 ps       TEMPERATURE             "
        "1.00000 K          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   PRESSURE                2.00000 bar      E(TOT)                 "
        "12.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               "
        "19.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             "
        "5.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   E(NH-MOMENTUM)         11.00000 kcal/mol E(NH-FRICTION)         "
        "12.00000 kcal/mol   |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                "
        "0.10000 s          |"
    );
    std::getline(file, line);
    EXPECT_EQ(
        line,
        "----------------------------------------------------------------------"
        "-------------------"
    );
}

/**
 * @brief tests writing info file
 *
 * @details qmmm is active: E(QM) is corrected by the QM reference energy,
 * which is printed next to N(SM-MOL)
 *
 */
TEST_F(TestEnergyOutput, writeInfoQmmmSubtractsQMReferenceEnergy)
{
    _physicalData->reset();

    _physicalData->set_qmEnergy(7.0 * kcal_per_mol);
    _physicalData->set_qmReferenceEnergy(2.0 * kcal_per_mol);
    _physicalData->set_numberOfSmoothingMol(1.0);

    _settings->forceFieldSettings.setTypeByUser(ForceFieldType::OFF);
    settings::GeneralSettings::setJobtype(JobType::QMMM_MD);
    _settings->manostatSettings.setManostatTypeByUser(ManostatType::NONE);
    _settings->thermostatSettings.setThermostatTypeByUser(ThermostatType::NONE);

    _infoOutput->setFilename("default.info", false);
    _infoOutput->write(100.0 * ps, *_physicalData, *_settings);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   content(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    settings::GeneralSettings::setJobtype(JobType::MM_MD);

    EXPECT_NE(
        content.find(
            "|   E(QM)                   5.00000 kcal/mol N(QM-ATOMS)       "
            "      0.00000 -          |"
        ),
        std::string::npos
    );
    EXPECT_NE(
        content.find(
            "|   N(SM-MOL)               1.00000 -        E(QM-REF)         "
            "      2.00000 kcal/mol   |"
        ),
        std::string::npos
    );
}
