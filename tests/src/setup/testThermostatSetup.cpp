#include "constants.hpp"
#include "exceptions.hpp"
#include "testSetup.hpp"
#include "thermostatSetup.hpp"

using namespace setup;
using namespace customException;

TEST_F(TestSetup, setupThermostat)
{
    ThermostatSetup thermostatSetup(_engine);

    _engine.getTimings().setTimestep(0.1);
    thermostatSetup.setup();

    EXPECT_EQ(_engine._thermostat->getTimestep(), 0.1);

    _engine.getSettings().setThermostat("berendsen");
    EXPECT_THROW(thermostatSetup.setup(), InputFileException);

    _engine.getSettings().setTemperature(300.0);
    EXPECT_NO_THROW(thermostatSetup.setup());

    thermostat::BerendsenThermostat *berendsenThermostat =
        dynamic_cast<thermostat::BerendsenThermostat *>(_engine._thermostat.get());
    EXPECT_EQ(berendsenThermostat->getTau(), 0.1 * 1000);

    _engine.getSettings().setRelaxationTime(0.2);
    EXPECT_NO_THROW(thermostatSetup.setup());

    auto *berendsenThermostat2 = dynamic_cast<thermostat::BerendsenThermostat *>(_engine._thermostat.get());
    EXPECT_EQ(berendsenThermostat2->getTau(), 0.2 * 1000);

    EXPECT_NO_THROW(setupThermostat(_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}