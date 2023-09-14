#include "testEnergyOutput.hpp"

#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "manostatSettings.hpp"     // for ManostatSettings
#include "settings.hpp"             // for Settings

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing energy output file
 *
 * @details minimal output data
 *
 */
TEST_F(TestEnergyOutput, forceFieldNotActive)
{
    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(6.0);
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    settings::ForceFieldSettings::deactivate();
    settings::Settings::activateMM();

    _energyOutput->setFilename("default.en");
    _energyOutput->write(100.0, 0.1, *_physicalData);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line,
              "       100\t      1.000000000000\t      2.000000000000\t     12.000000000000\t      3.000000000000\t     "
              "19.000000000000\t      4.000000000000\t      5.000000000000\t         6.00000e+00\t     0.10000");
}

/**
 * @brief tests writing energy output file
 *
 * @details force field is set
 *
 */
TEST_F(TestEnergyOutput, forceFieldActive)
{
    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(6.0);
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    _physicalData->setBondEnergy(19.0);
    _physicalData->setAngleEnergy(20.0);
    _physicalData->setDihedralEnergy(21.0);
    _physicalData->setImproperEnergy(22.0);

    settings::ForceFieldSettings::activate();
    settings::Settings::activateMM();

    _energyOutput->setFilename("default.en");
    _energyOutput->write(100.0, 0.1, *_physicalData);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line,
              "       100\t      1.000000000000\t      2.000000000000\t     94.000000000000\t      3.000000000000\t     "
              "19.000000000000\t      4.000000000000\t      5.000000000000\t     19.000000000000\t     20.000000000000\t     "
              "21.000000000000\t     22.000000000000\t         6.00000e+00\t     0.10000");
}

/**
 * @brief tests writing energy output file
 *
 * @details manostat is set
 *
 */
TEST_F(TestEnergyOutput, manostatActive)
{
    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setCoulombEnergy(4.0);
    _physicalData->setNonCoulombEnergy(5.0);
    _physicalData->setMomentum(6.0);
    _physicalData->setIntraCoulombEnergy(9.0);
    _physicalData->setIntraNonCoulombEnergy(10.0);

    _physicalData->setVolume(19.0);
    _physicalData->setDensity(20.0);

    settings::ForceFieldSettings::deactivate();
    settings::ManostatSettings::setManostatType("Berendsen");
    settings::Settings::activateMM();

    _energyOutput->setFilename("default.en");
    _energyOutput->write(100.0, 0.1, *_physicalData);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line,
              "       100\t      1.000000000000\t      2.000000000000\t     12.000000000000\t      3.000000000000\t     "
              "19.000000000000\t      4.000000000000\t      5.000000000000\t     19.000000000000\t     20.000000000000\t   "
              "      6.00000e+00\t     0.10000");
}

/**
 * @brief tests writing energy output file
 *
 * @details qm is active is set
 *
 */
TEST_F(TestEnergyOutput, qmActive)
{
    _physicalData->reset();

    _physicalData->setTemperature(1.0);
    _physicalData->setPressure(2.0);
    _physicalData->setKineticEnergy(3.0);
    _physicalData->setMomentum(6.0);
    _physicalData->setIntraCoulombEnergy(0.0);
    _physicalData->setIntraNonCoulombEnergy(0.0);

    _physicalData->setQMEnergy(5.0);

    _physicalData->setVolume(19.0);
    _physicalData->setDensity(20.0);

    settings::ForceFieldSettings::deactivate();
    settings::Settings::activateQM();
    settings::Settings::deactivateMM();
    settings::ManostatSettings::setManostatType("none");

    _energyOutput->setFilename("default.en");
    _energyOutput->write(100.0, 0.1, *_physicalData);
    _energyOutput->close();

    std::ifstream file("default.en");
    std::string   line;
    std::getline(file, line);
    EXPECT_EQ(line,
              "       100\t      1.000000000000\t      2.000000000000\t      8.000000000000\t      5.000000000000\t      "
              "0.000000000000\t      3.000000000000\t      0.000000000000\t         6.00000e+00\t     0.10000");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}