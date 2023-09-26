#include "qmmdEngine.hpp"

#include "integrator.hpp"      // for Integrator
#include "manostat.hpp"        // for Manostat
#include "physicalData.hpp"    // for PhysicalData
#include "resetKinetics.hpp"   // for ResetKinetics
#include "thermostat.hpp"      // for Thermostat

using engine::QMMDEngine;

/**
 * @brief Takes one step in a QM MD simulation.
 *
 * @details The step is taken in the following order:
 * - First step of the integrator
 * - Apply thermostat half step
 * - Run QM calculations
 * - Apply thermostat on forces
 * - Second step of the integrator
 * - Apply thermostat
 * - Calculate kinetic energy and momentum
 * - Apply manostat
 * - Reset temperature and momentum
 *
 */
void QMMDEngine::takeStep()
{
    _thermostat->applyThermostatHalfStep(_simulationBox, _physicalData);

    _integrator->firstStep(_simulationBox);

    _qmRunner->run(_simulationBox, _physicalData);

    _thermostat->applyThermostatOnForces(_simulationBox);

    _integrator->secondStep(_simulationBox);

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKinetics(_simulationBox);

    _manostat->applyManostat(_simulationBox, _physicalData);

    _resetKinetics.reset(_step, _physicalData, _simulationBox);
}