#include "ringPolymerqmmdEngine.hpp"

using engine::RingPolymerQMMDEngine;

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
        _integrator->secondStep(bead);

        _thermostat->applyThermostat(bead, _physicalData);

        _physicalData.calculateKineticEnergyAndMomentum(bead);

        _manostat->applyManostat(bead, _physicalData);

        _resetKinetics->reset(_step, _physicalData, bead);
    };

    std::ranges::for_each(_ringPolymerBeads, afterRingPolymerCoupling);

    combineBeads();
}