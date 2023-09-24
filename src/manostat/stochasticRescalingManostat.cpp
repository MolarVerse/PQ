#include "stochasticRescalingManostat.hpp"

#include "constants.hpp"            // for Constants
#include "physicalData.hpp"         // for PhysicalData
#include "simulationBox.hpp"        // for SimulationBox
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "timingsSettings.hpp"      // for TimingsSettings

using manostat::StochasticRescalingManostat;

/**
 * @brief copy constructor
 *
 * @param other
 */
StochasticRescalingManostat::StochasticRescalingManostat(const StochasticRescalingManostat &other)
    : Manostat(other), _tau(other._tau), _compressibility(other._compressibility){};

/**
 * @brief apply Stochastic Rescaling manostat for NPT ensemble
 *
 * @param simBox
 * @param physicalData
 */
void StochasticRescalingManostat::applyManostat(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData)
{
    calculatePressure(simBox, physicalData);

    const auto compressibilityFactor = _compressibility * settings::TimingsSettings::getTimeStep() / _tau;

    const auto stochasticFactor =
        ::sqrt(2.0 * constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ * settings::ThermostatSettings::getTargetTemperature() *
               compressibilityFactor / (simBox.getVolume()) * constants::_PRESSURE_FACTOR_) *
        std::normal_distribution<double>(0.0, 1.0)(_generator);

    const auto linearScalingFactor =
        ::pow(exp(-compressibilityFactor * (_targetPressure - _pressure) + stochasticFactor), 1.0 / 3.0);

    const auto scalingFactors = linearAlgebra::Vec3D(linearScalingFactor);

    simBox.scaleBox(scalingFactors);

    physicalData.setVolume(simBox.getVolume());
    physicalData.setDensity(simBox.getDensity());

    simBox.checkCoulombRadiusCutOff(customException::ExceptionType::MANOSTATEXCEPTION);

    auto scalePositions  = [&scalingFactors](auto &molecule) { molecule.scale(scalingFactors); };
    auto scaleVelocities = [&scalingFactors](auto &atom) { atom->scaleVelocity(1.0 / scalingFactors); };

    std::ranges::for_each(simBox.getMolecules(), scalePositions);
    std::ranges::for_each(simBox.getAtoms(), scaleVelocities);
}