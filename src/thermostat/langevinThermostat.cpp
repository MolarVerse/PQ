#include "langevinThermostat.hpp"

#include "constants.hpp"   // for constants::_BOLTZMANN_CONSTANT_

#include <cmath>   // for std::sqrt

using thermostat::LangevinThermostat;

LangevinThermostat::LangevinThermostat(const double targetTemperature) : Thermostat(targetTemperature)
{
    _sigma = std::sqrt(4.0 * constants::_BOLTZMANN_CONSTANT_ * targetTemperature / 1.0e-12);
}