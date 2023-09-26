#include "ringPolymerqmmdEngine.hpp"

#include "integrator.hpp"      // for Integrator
#include "manostat.hpp"        // for Manostat
#include "physicalData.hpp"    // for PhysicalData
#include "qmRunner.hpp"        // for QMRunner
#include "resetKinetics.hpp"   // for ResetKinetics
#include "thermostat.hpp"      // for Thermostat

#include <algorithm>    // for __for_each_fn, for_each
#include <functional>   // for identity
#include <memory>       // for unique_ptr

using engine::RingPolymerQMMDEngine;

/**
 * @brief Takes one step in a ring polymer QM MD simulation.
 *
 * @details The step is taken in the following order:
 * - First step of the integrator
 * - Apply thermostat half step
 * - Run QM calculations
 * - couple ring polymer beads
 * - Apply thermostat on forces
 * - Second step of the integrator
 * - Apply thermostat
 * - Calculate kinetic energy and momentum
 * - Apply manostat
 * - Reset temperature and momentum
 *
 */
void RingPolymerQMMDEngine::takeStep()
{
    auto beforeRingPolymerCoupling = [this](auto &bead)
    {
        _thermostat->applyThermostatHalfStep(_simulationBox, _physicalData);

        _integrator->firstStep(bead);

        _qmRunner->run(bead, _physicalData);
    };

    std::ranges::for_each(_ringPolymerBeads, beforeRingPolymerCoupling);

    coupleRingPolymerBeads();

    auto afterRingPolymerCoupling = [this](auto &bead)
    {
        _thermostat->applyThermostatOnForces(bead);

        _integrator->secondStep(bead);

        _thermostat->applyThermostat(bead, _physicalData);

        _physicalData.calculateKinetics(bead);

        _manostat->applyManostat(bead, _physicalData);

        _resetKinetics.reset(_step, _physicalData, bead);
    };

    std::ranges::for_each(_ringPolymerBeads, afterRingPolymerCoupling);

    combineBeads();
}