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
#include "staticMatrix3x3.hpp"               // for operator*, operator+=
#include "staticMatrix3x3Class.hpp"          // for StaticMatrix3x3
#include "thermostatSettings.hpp"            // for ThermostatSettings
#include "vector3d.hpp"                      // for Vec3D, Vector3D, cross

using namespace resetKinetics;
using linearAlgebra::diagonalMatrix;
using linearAlgebra::inverse;
using linearAlgebra::StaticMatrix3x3;
using linearAlgebra::tensorProduct;
using linearAlgebra::trace;

/**
 * @brief checks to reset angular momentum
 *
 * @param step
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::reset(
    const size_t                  step,
    physicalData::PhysicalData   &physicalData,
    simulationBox::SimulationBox &simBox
)
{
    startTimingsSection("Reset Kinetics");

    _momentum        = physicalData.getMomentum() * constants::_S_TO_FS_;
    _angularMomentum = physicalData.getAngularMomentum() * constants::_S_TO_FS_;
    _temperature     = physicalData.getTemperature();

    const bool resetTemperature = (step <= _nStepsTemperatureReset) ||
                                  (0 == step % _frequencyTemperatureReset);
    const bool resetMomentum =
        !resetTemperature && ((step <= _nStepsMomentumReset) ||
                              (0 == step % _frequencyMomentumReset));
    const bool resetAngular =
        (step <= _nStepsAngularReset) || (0 == step % _frequencyAngularReset);

    if (resetTemperature)
    {
        ResetKinetics::resetTemperature(simBox);
        ResetKinetics::resetMomentum(simBox);
    }

    if (resetMomentum)
        ResetKinetics::resetMomentum(simBox);

    if (resetAngular)
        ResetKinetics::resetAngularMomentum(simBox);

    physicalData.setTemperature(_temperature);
    physicalData.setMomentum(_momentum * constants::_FS_TO_S_);
    physicalData.setAngularMomentum(_angularMomentum * constants::_FS_TO_S_);

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
void ResetKinetics::resetTemperature(simulationBox::SimulationBox &simBox)
{
    const auto targetTemperature =
        settings::ThermostatSettings::getTargetTemperature();
    const auto lambda = ::sqrt(targetTemperature / _temperature);

    std::ranges::for_each(
        simBox.getAtoms(),
        [lambda](auto &atom) { atom->scaleVelocity(lambda); }
    );

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
void ResetKinetics::resetMomentum(simulationBox::SimulationBox &simBox)
{
    const auto momentumVector     = _momentum;
    const auto momentumCorrection = momentumVector / simBox.getTotalMass();

    std::ranges::for_each(
        simBox.getAtoms(),
        [momentumCorrection](auto &atom)
        { atom->addVelocity(-momentumCorrection); }
    );

    _temperature     = simBox.calculateTemperature();
    _momentum        = simBox.calculateMomentum();
    _angularMomentum = simBox.calculateAngularMomentum(_momentum);
}

/**
 * @brief reset the angular momentum of the system
 *
 * @details subtract angular momentum correction from all velocities -
 * correction is the total angular momentum divided by the total mass
 *
 * @param physicalData
 * @param simBox
 */
void ResetKinetics::resetAngularMomentum(simulationBox::SimulationBox &simBox)
{
    simBox.calculateCenterOfMass();
    const auto centerOfMass = simBox.getCenterOfMass();

    _angularMomentum = simBox.calculateAngularMomentum(_momentum);

    StaticMatrix3x3 helperMatrix{0.0};

    auto addInertiaTensorOfAtom =
        [&helperMatrix, &centerOfMass](const auto &atom)
    {
        auto relativePosition = atom->getPosition() - centerOfMass;
        helperMatrix +=
            tensorProduct(relativePosition, relativePosition) * atom->getMass();
    };

    std::ranges::for_each(simBox.getAtoms(), addInertiaTensorOfAtom);

    const StaticMatrix3x3 inertiaTensor =
        -helperMatrix + diagonalMatrix(trace(helperMatrix));
    const StaticMatrix3x3 inverseInertiaTensor = inverse(inertiaTensor);

    const auto angularVelocity = inverseInertiaTensor * _angularMomentum;

    auto correctVelocities = [&angularVelocity, &centerOfMass](auto &atom)
    {
        auto relativePosition = atom->getPosition() - centerOfMass;
        atom->addVelocity(-cross(angularVelocity, relativePosition));
    };

    std::ranges::for_each(simBox.getAtoms(), correctVelocities);

    _temperature     = simBox.calculateTemperature();
    _momentum        = simBox.calculateMomentum();
    _angularMomentum = simBox.calculateAngularMomentum(_momentum);
}