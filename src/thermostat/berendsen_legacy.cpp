#include "berendsenThermostat.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"

using namespace thermostat;
using namespace physicalData;
using namespace simulationBox;
using namespace settings;

/**
 * @brief apply thermostat - Berendsen
 *
 * @link https://doi.org/10.1063/1.448118
 *
 * @param simulationBox
 * @param data
 */
void BerendsenThermostat::applyThermostat(
    SimulationBox &simulationBox,
    PhysicalData  &data
)
{
    startTimingsSection("Berendsen");

    _temperature = simulationBox.calculateTemperature();

    const auto dt        = TimingsSettings::getTimeStep();
    const auto tempRatio = _targetTemperature / _temperature;

    const auto berendsenFactor = ::sqrt(1.0 + dt / _tau * (tempRatio - 1.0));

    for (const auto &atom : simulationBox.getAtoms())
        atom->scaleVelocity(berendsenFactor);

    data.setTemperature(_temperature * berendsenFactor * berendsenFactor);

    stopTimingsSection("Berendsen");
}