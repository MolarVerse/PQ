#include "constants.hpp"
#include "exceptions.hpp"
#include "testSetup.hpp"
#include "thermostatSettings.hpp"
#include "thermostatSetup.hpp"

using namespace setup;
using namespace customException;

TEST_F(TestSetup, setupThermostat)
{
    ThermostatSetup thermostatSetup(_engine);

    _engine.getTimings().setTimestep(0.1);
    thermostatSetup.setup();

    EXPECT_EQ(_engine.getThermostat().getTimestep(), 0.1);

    settings::ThermostatSettings::setThermostatType("berendsen");
    EXPECT_THROW(thermostatSetup.setup(), InputFileException);

    settings::ThermostatSettings::setTargetTemperature(300);
    settings::ThermostatSettings::setTemperatureSet(true);
    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto berendsenThermostat = dynamic_cast<thermostat::BerendsenThermostat &>(_engine.getThermostat());
    EXPECT_EQ(berendsenThermostat.getTau(), 0.1 * 1000);

    settings::ThermostatSettings::setRelaxationTime(0.2);
    EXPECT_NO_THROW(thermostatSetup.setup());

    const auto berendsenThermostat2 = dynamic_cast<thermostat::BerendsenThermostat &>(_engine.getThermostat());
    EXPECT_EQ(berendsenThermostat2.getTau(), 0.2 * 1000);

    EXPECT_NO_THROW(setupThermostat(_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}