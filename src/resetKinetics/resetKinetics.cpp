/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "resetKinetics.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <cmath>        // for sqrt
#include <cstddef>      // for size_t
#include <functional>   // for identity

#include "constants/conversionFactors.hpp"   // for _FS_TO_S_, _S_TO_FS_
#include "physicalData.hpp"                  // for PhysicalData
#include "simulationBox.hpp"                 // for SimulationBox
#include "staticMatrix.hpp"                  // for operator*, operator+=
#include "thermostatSettings.hpp"            // for ThermostatSettings
#include "vector3d.hpp"                      // for Vec3D, Vector3D, cross

using namespace resetKinetics;
using namespace linearAlgebra;
using namespace physicalData;
using namespace simulationBox;
using namespace constants;
using namespace settings;

/**
 * @brief Construct a new Reset Kinetics:: Reset Kinetics object
 *
 * @param nStepsTemperatureReset
 * @param frequencyTemperatureReset
 * @param nStepsMomentumReset
 * @param frequencyMomentumReset
 * @param nStepsAngularReset
 * @param frequencyAngularReset
 */
ResetKinetics::ResetKinetics(
    const size_t nStepsTemperatureReset,
    const size_t frequencyTemperatureReset,
    const size_t nStepsMomentumReset,
    const size_t frequencyMomentumReset,
    const size_t nStepsAngularReset,
    const size_t frequencyAngularReset
)
    : _nStepsTemperatureReset(nStepsTemperatureReset),
      _frequencyTemperatureReset(frequencyTemperatureReset),
      _nStepsMomentumReset(nStepsMomentumReset),
      _frequencyMomentumReset(frequencyMomentumReset),
      _nStepsAngularReset(nStepsAngularReset),
      _frequencyAngularReset(frequencyAngularReset)
{
}

/**
 * @brief checks to reset angular momentum
 *
 * @param step
 * @param data
 * @param simBox
 */
void ResetKinetics::reset(
    const size_t   step,
    PhysicalData  &data,
    SimulationBox &simBox
)
{
    startTimingsSection("Reset Kinetics");

    _momentum        = data.getMomentum() * _S_TO_FS_;
    _angularMomentum = data.getAngularMomentum() * _S_TO_FS_;
    _temperature     = data.getTemperature();

    auto resetTemp = (step <= _nStepsTemperatureReset);
    resetTemp      = resetTemp || (0 == step % _frequencyTemperatureReset);

    auto resetMom = (step <= _nStepsMomentumReset);
    resetMom      = resetMom || (0 == step % _frequencyMomentumReset);
    resetMom      = !resetTemp && resetMom;

    auto resetAngular = (step <= _nStepsAngularReset);
    resetAngular      = resetAngular || (0 == step % _frequencyAngularReset);

    if (resetTemp)
    {
        ResetKinetics::resetTemperature(simBox);
        ResetKinetics::resetMomentum(simBox);
    }

    if (resetMom)
        ResetKinetics::resetMomentum(simBox);

    if (resetAngular)
        ResetKinetics::resetAngularMomentum(simBox);

    data.setTemperature(_temperature);
    data.setMomentum(_momentum * _FS_TO_S_);
    data.setAngularMomentum(_angularMomentum * _FS_TO_S_);

    stopTimingsSection("Reset Kinetics");
}

/**
 * @brief reset the temperature of the system - hard scaling
 *
 * @details calculate hard scaling factor for target temperature and current
 * temperature and scale all velocities
 *
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::resetTemperature(SimulationBox &simBox)
{
    const auto targetTemp = ThermostatSettings::getActualTargetTemperature();
    const auto lambda     = ::sqrt(targetTemp / _temperature);

    simBox.scaleVelocities(lambda);

    _temperature     = simBox.calculateTemperature();
    _momentum        = simBox.calculateMomentum();
    _angularMomentum = simBox.calculateAngularMomentum(_momentum);
}

/**
 * @brief reset the momentum of the system
 *
 * @details subtract momentum correction from all velocities - correction is the
 * total momentum divided by the total mass
 *
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::resetMomentum(SimulationBox &simBox)
{
    const auto momentumVector     = _momentum;
    const auto momentumCorrection = momentumVector / simBox.getTotalMass();

    simBox.addToVelocities(-momentumCorrection);

    _temperature     = simBox.calculateTemperature();
    _momentum        = simBox.calculateMomentum();
    _angularMomentum = simBox.calculateAngularMomentum(_momentum);
}

/********************
 *                  *
 * standard setters *
 *                  *
 *******************/

/**
 * @brief set the temperature
 *
 * @param temperature
 */
void ResetKinetics::setTemperature(const double temperature)
{
    _temperature = temperature;
}

/**
 * @brief set the momentum
 *
 * @param momentum
 */
void ResetKinetics::setMomentum(const pq::Vec3D &momentum)
{
    _momentum = momentum;
}

/**
 * @brief set the angular momentum
 *
 * @param angularMomentum
 */
void ResetKinetics::setAngularMomentum(const pq::Vec3D &angularMomentum)
{
    _angularMomentum = angularMomentum;
}

/********************
 *                  *
 * standard getters *
 *                  *
 *******************/

/**
 * @brief get the number of steps for temperature reset
 *
 * @return size_t
 */
size_t ResetKinetics::getNStepsTemperatureReset() const
{
    return _nStepsTemperatureReset;
}

/**
 * @brief get the frequency for temperature reset
 *
 * @return size_t
 */
size_t ResetKinetics::getFrequencyTemperatureReset() const
{
    return _frequencyTemperatureReset;
}

/**
 * @brief get the number of steps for momentum reset
 *
 * @return size_t
 */
size_t ResetKinetics::getNStepsMomentumReset() const
{
    return _nStepsMomentumReset;
}

/**
 * @brief get the frequency for momentum reset
 *
 * @return size_t
 */
size_t ResetKinetics::getFrequencyMomentumReset() const
{
    return _frequencyMomentumReset;
}