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

    simulationBox.flattenVelocities();
    auto *const velPtr = simulationBox.getVelPtr();

    // clang-format off
    #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(velPtr)
    for (size_t i = 0; i < simulationBox.getNumberOfAtoms(); ++i)
        for (size_t j = 0; j < 3; ++j)
            velPtr[i*3 + j] *= berendsenFactor;

    // clang-format on

    simulationBox.deFlattenVelocities();

    data.setTemperature(_temperature * berendsenFactor * berendsenFactor);

    stopTimingsSection("Berendsen");
}