#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "infoOutput.hpp"           // for InfoOutput
#include "manostatSettings.hpp"     // for ManostatSettings
#include "physicalData.hpp"         // for PhysicalData
#include "settings.hpp"             // for Settings
#include "testEnergyOutput.hpp"     // for TestEnergyOutput
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "vector3d.hpp"             // for Vec3D

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, InitGoogleTest, RUN_ALL_TESTS
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing info file
 *
 * @details basic info file
 *
 */
TEST_F(TestEnergyOutput, writeInfo_forceFieldNotActive)
{
    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linearAlgebra::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    settings::ForceFieldSettings::deactivate();
    settings::Settings::activateMM();

    _infoOutput->setFilename("default.info");
    _infoOutput->write(100.0, 0.1, *_physicalData);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|                                  PIMD-QMCF info file                                  |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|   SIMULATION TIME       100.00000 ps       TEMPERATURE             1.00000 K          |");
    getline(file, line);
    EXPECT_EQ(line, "|   PRESSURE                2.00000 bar      E(TOT)                 12.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               19.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             5.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                0.10000 s          |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
}

/**
 * @brief tests writing info file
 *
 * @details force field is active
 *
 */
TEST_F(TestEnergyOutput, writeInfo_forceFieldActive)
{
    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linearAlgebra::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    _physicalData->setBondEnergy(19.0);
    _physicalData->setAngleEnergy(20.0);
    _physicalData->setDihedralEnergy(21.0);
    _physicalData->setImproperEnergy(22.0);

    settings::ForceFieldSettings::activate();
    settings::Settings::activateMM();

    _infoOutput->setFilename("default.info");
    _infoOutput->write(100.0, 0.1, *_physicalData);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|                                  PIMD-QMCF info file                                  |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|   SIMULATION TIME       100.00000 ps       TEMPERATURE             1.00000 K          |");
    getline(file, line);
    EXPECT_EQ(line, "|   PRESSURE                2.00000 bar      E(TOT)                 94.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               19.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             5.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(BOND)                19.00000 kcal/mol E(ANGLE)               20.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(DIHEDRAL)            21.00000 kcal/mol E(IMPROPER)            22.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                0.10000 s          |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
}

/**
 * @brief tests writing info file
 *
 * @details manostat is active
 *
 */
TEST_F(TestEnergyOutput, writeInfo_manostatIsActive)
{
    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linearAlgebra::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    _physicalData->setVolume(11.0);
    _physicalData->setDensity(12.0);

    settings::ForceFieldSettings::deactivate();
    settings::ManostatSettings::setManostatType("Berendsen");
    settings::Settings::activateMM();

    _infoOutput->setFilename("default.info");
    _infoOutput->write(100.0, 0.1, *_physicalData);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|                                  PIMD-QMCF info file                                  |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|   SIMULATION TIME       100.00000 ps       TEMPERATURE             1.00000 K          |");
    getline(file, line);
    EXPECT_EQ(line, "|   PRESSURE                2.00000 bar      E(TOT)                 12.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               19.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             5.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   VOLUME                 11.00000 A^3      DENSITY                12.00000 g/cm^3     |");
    getline(file, line);
    EXPECT_EQ(line, "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                0.10000 s          |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
}

/**
 * @brief tests writing info file
 *
 * @details qm is active
 *
 */
TEST_F(TestEnergyOutput, writeInfo_qmIsActive)
{
    _physicalData->reset();

    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setMomentum(linearAlgebra::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(0.0);
    _physicalData->setIntraNonCoulombEnergy(0.0);

    _physicalData->setQMEnergy(5.0);

    _physicalData->setVolume(19.0);
    _physicalData->setDensity(20.0);

    settings::ForceFieldSettings::deactivate();
    settings::Settings::activateQM();
    settings::Settings::deactivateMM();
    settings::ManostatSettings::setManostatType("none");

    _infoOutput->setFilename("default.info");
    _infoOutput->write(100.0, 0.1, *_physicalData);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|                                  PIMD-QMCF info file                                  |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|   SIMULATION TIME       100.00000 ps       TEMPERATURE             1.00000 K          |");
    getline(file, line);
    EXPECT_EQ(line, "|   PRESSURE                2.00000 bar      E(TOT)                  8.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(QM)                   5.00000 kcal/mol N(QM ATOMS)             0.00000            |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(KIN)                  3.00000 kcal/mol E(INTRA)                0.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                0.10000 s          |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
}

/**
 * @brief tests writing info file
 *
 * @details ring polymer is active
 *
 */
TEST_F(TestEnergyOutput, writeInfo_ringPolymerActive)
{
    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linearAlgebra::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);
    _physicalData->setRingPolymerEnergy({1.0, 2.0});

    settings::ForceFieldSettings::deactivate();
    settings::Settings::activateMM();
    settings::Settings::deactivateQM();
    settings::Settings::activateRingPolymerMD();
    settings::ManostatSettings::setManostatType("none");

    _infoOutput->setFilename("default.info");
    _infoOutput->write(100.0, 0.1, *_physicalData);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|                                  PIMD-QMCF info file                                  |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|   SIMULATION TIME       100.00000 ps       TEMPERATURE             1.00000 K          |");
    getline(file, line);
    EXPECT_EQ(line, "|   PRESSURE                2.00000 bar      E(TOT)                 12.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(MEAN RPMD)            1.50000 kcal/mol E(MAX RPMD)             2.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               19.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             5.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                0.10000 s          |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
}

/**
 * @brief tests writing info file
 *
 * @details nose hoover is active
 *
 */
TEST_F(TestEnergyOutput, writeInfo_NoseHooverActive)
{
    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(linearAlgebra::Vec3D(6.0));
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);
    _physicalData->setNoseHooverMomentumEnergy(11.0);
    _physicalData->setNoseHooverFrictionEnergy(12.0);

    settings::ForceFieldSettings::deactivate();
    settings::Settings::activateMM();
    settings::Settings::deactivateQM();
    settings::Settings::deactivateRingPolymerMD();
    settings::ManostatSettings::setManostatType("none");
    settings::ThermostatSettings::setThermostatType(settings::ThermostatType::NOSE_HOOVER);

    _infoOutput->setFilename("default.info");
    _infoOutput->write(100.0, 0.1, *_physicalData);
    _infoOutput->close();

    std::ifstream file("default.info");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|                                  PIMD-QMCF info file                                  |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
    getline(file, line);
    EXPECT_EQ(line, "|   SIMULATION TIME       100.00000 ps       TEMPERATURE             1.00000 K          |");
    getline(file, line);
    EXPECT_EQ(line, "|   PRESSURE                2.00000 bar      E(TOT)                 12.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(KIN)                  3.00000 kcal/mol E(INTRA)               19.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(COUL)                 4.00000 kcal/mol E(NON-COUL)             5.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   E(NH MOMENTUM)         11.00000 kcal/mol E(NH FRICTION)         12.00000 kcal/mol   |");
    getline(file, line);
    EXPECT_EQ(line, "|   MOMENTUM                1.0e+01 amuA/fs  LOOPTIME                0.10000 s          |");
    getline(file, line);
    EXPECT_EQ(line, "-----------------------------------------------------------------------------------------");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}