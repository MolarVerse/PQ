#include "noseHooverThermostat.hpp"

#include "constants.hpp"         // for _FS_TO_S_
#include "physicalData.hpp"      // for PhysicalData
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for TimingsSettings

#include <algorithm>   // for for_each

using thermostat::NoseHooverThermostat;

/**
 * @brief applies the Nose-Hoover thermostat on the forces
 *
 * @details the Nose-Hoover thermostat is applied on the forces of the atoms after force calculation
 *
 * @param simBox simulation box
 */
void NoseHooverThermostat::applyThermostatOnForces(simulationBox::SimulationBox &simBox)
{
    const auto   kT_target        = constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * _targetTemperature;
    const double degreesOfFreedom = simBox.getDegreesOfFreedom();
    const auto   factor =
        _chi[0] * _couplingFrequency * _couplingFrequency / (kT_target * degreesOfFreedom) * constants::_MOMENTUM_TO_FORCE_;

    auto applyNoseHoover = [factor](auto &atom) { atom->addForce(-factor * atom->getVelocity() * atom->getMass()); };

    std::ranges::for_each(simBox.getAtoms(), applyNoseHoover);
}

void NoseHooverThermostat::applyThermostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData)
{
    physicalData.calculateTemperature(simBox);

    _temperature                  = physicalData.getTemperature();
    const double degreesOfFreedom = simBox.getDegreesOfFreedom();
    const auto   timestep         = settings::TimingsSettings::getTimeStep() * constants::_FS_TO_S_;
    const auto   kT               = constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * _temperature;
    const auto   kT_target        = constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * _targetTemperature;

    _chi[0] += timestep *
               ((kT - kT_target) * degreesOfFreedom - _chi[0] * _chi[1] / kT_target * _couplingFrequency * _couplingFrequency);
    auto ratio  = _chi[0] / (kT_target * degreesOfFreedom) * _couplingFrequency * _couplingFrequency;
    _zeta[0]   += ratio * timestep;
    ratio      *= _chi[0];

    auto energyMomentum = ratio;
    auto energyFriction = degreesOfFreedom * _zeta[0];

    for (size_t i = 1; i < _chi.size() - 1; ++i)
    {
        _chi[i]  += timestep * (ratio - kT_target - _chi[i] * _chi[i + 1] / kT_target * _couplingFrequency * _couplingFrequency);
        ratio     = _chi[i] / kT_target * _couplingFrequency * _couplingFrequency;
        _zeta[i] += ratio * timestep;
        ratio    *= _chi[i];

        energyMomentum += ratio;
        energyFriction += _zeta[i];
    }

    physicalData.setNoseHooverMomentumEnergy(energyMomentum);
    physicalData.setNoseHooverFrictionEnergy(energyFriction);
}