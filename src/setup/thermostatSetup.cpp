#include "thermostatSetup.hpp"

#include "constants.hpp"
#include "exceptions.hpp"

using namespace std;
using namespace setup;
using namespace thermostat;
using namespace engine;

/**
 * @brief wrapper for thermostat setup
 *
 * @param engine
 */
void setup::setupThermostat(Engine &engine)
{
    ThermostatSetup thermostatSetup(engine);
    thermostatSetup.setup();
}

/**
 * @brief setup thermostat
 *
 * TODO: include warnings if value set but not used
 *
 */
void ThermostatSetup::setup()
{
    if (_engine.getSettings().getThermostat() == "berendsen")
    {
        if (!_engine.getSettings().getTemperatureSet())
            throw customException::InputFileException("Temperature not set for Berendsen thermostat");

        if (!_engine.getSettings().getRelaxationTimeSet())
        {
            _engine._stdoutOutput->writeRelaxationTimeThermostatWarning();
            _engine._logOutput->writeRelaxationTimeThermostatWarning();
        }

        _engine._thermostat = make_unique<BerendsenThermostat>(_engine.getSettings().getTemperature(),
                                                               _engine.getSettings().getRelaxationTime() * config::_PS_TO_FS_);
    }
    else
    {
        // warnings if values set but not used
    }

    _engine._thermostat->setTimestep(_engine.getTimings().getTimestep());
}