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

#include <algorithm>   // for __for_each_fn, for_each
#include <cmath>       // for sqrt
#include <cstddef>     // for size_t

#include "constants/conversionFactors.hpp"   // for _FS_TO_S_, _S_TO_FS_
#include "exceptions.hpp"                    // for UserInputException
#include "globalTimer.hpp"
#include "mathUtilities.hpp"        // for isZero
#include "physicalData.hpp"         // for PhysicalData
#include "simulationBox.hpp"        // for SimulationBox
#include "staticMatrix.hpp"         // for operator*, operator+=
#include "thermostatSettings.hpp"   // for ThermostatSettings
#include "vector3d.hpp"             // for Vec3D, Vector3D, cross

using namespace resetKinetics;
using namespace linearAlgebra;
using namespace physicalData;
using namespace molsys;
using namespace constants;
using namespace exc;
using namespace settings;
using namespace utilities;

/**
 * @brief Construct a new Reset Kinetics:: Reset Kinetics object
 *
 * @param nStepsTemperatureReset
 * @param frequencyTemperatureReset
 * @param nStepsMomentumReset
 * @param frequencyMomentumReset
 * @param nStepsAngularReset
 * @param frequencyAngularReset
 * @param nStepsForcesReset
 */
ResetKinetics::ResetKinetics(
    size_t nStepsTemperatureReset,
    size_t frequencyTemperatureReset,
    size_t nStepsMomentumReset,
    size_t frequencyMomentumReset,
    size_t nStepsAngularReset,
    size_t frequencyAngularReset,
    size_t nStepsForcesReset
)
    : _nStepsTemperatureReset(nStepsTemperatureReset),
      _frequencyTemperatureReset(frequencyTemperatureReset),
      _nStepsMomentumReset(nStepsMomentumReset),
      _frequencyMomentumReset(frequencyMomentumReset),
      _nStepsAngularReset(nStepsAngularReset),
      _frequencyAngularReset(frequencyAngularReset),
      _nStepsForcesReset(nStepsForcesReset)
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
    size_t         step,
    PhysicalData  &data,
    SimulationBox &simBox
) const
{
    auto _ = scopedTimer(TimerId::ResetKinetics, "Reset Kinetics");

    auto momentum        = data.getMomentum() * S_TO_FS;
    auto angularMomentum = data.getAngularMomentum() * S_TO_FS;
    auto temperature     = data.getTemperature();

    auto resetTemp = (step <= _nStepsTemperatureReset);
    resetTemp      = resetTemp || (0 == step % _frequencyTemperatureReset);

    auto resetMom = (step <= _nStepsMomentumReset);
    resetMom      = resetMom || (0 == step % _frequencyMomentumReset);

    auto resetAngular = (step <= _nStepsAngularReset);
    resetAngular      = resetAngular || (0 == step % _frequencyAngularReset);

    if (resetTemp)
    {
        resetTemperature(simBox, temperature);
        momentum = simBox.calculateMomentum();
        resetMomentum(simBox, momentum);
        temperature     = simBox.calculateTemperature();
        momentum        = simBox.calculateMomentum();
        angularMomentum = simBox.calculateAngularMomentum(momentum);
    }
    else if (resetMom)
    {
        // temperature also needs reset of momentum, thus the else if
        ResetKinetics::resetMomentum(simBox, momentum);
        momentum        = simBox.calculateMomentum();
        temperature     = simBox.calculateTemperature();
        angularMomentum = simBox.calculateAngularMomentum(momentum);
    }

    if (resetAngular)
    {
        ResetKinetics::resetAngularMomentum(simBox, angularMomentum);
        temperature     = simBox.calculateTemperature();
        momentum        = simBox.calculateMomentum();
        angularMomentum = simBox.calculateAngularMomentum(momentum);
    }

    data.setTemperature(temperature);
    data.setMomentum(momentum * FS_TO_S);
    data.setAngularMomentum(angularMomentum * FS_TO_S);
}

/**
 * @brief reset the temperature of the system - hard scaling
 *
 * @details calculate hard scaling factor for target temperature and current
 * temperature and scale all velocities
 *
 * @param simBox
 * @param temperature
 */
void ResetKinetics::resetTemperature(SimulationBox &simBox, double temperature)
{
    const auto targetTemp = ThermostatSettings::getActualTargetTemperature();

    if (isZero(temperature))
    {
        throw UserInputException(
            "Cannot rescale a zero-temperature system. Initialize velocities "
            "first."
        );
    }

    const auto lambda = ::sqrt(targetTemp / temperature);

    std::ranges::for_each(
        simBox.getAtoms(),
        [lambda](auto &atom) { atom->scaleVelocity(lambda); }
    );
}

/**
 * @brief reset the momentum of the system
 *
 * @details subtract momentum correction from all velocities - correction is the
 * total momentum divided by the total mass
 *
 * @param simBox
 */
void ResetKinetics::resetMomentum(SimulationBox &simBox, const Vec3D &momentum)
{
    const auto momentumCorrection = momentum / simBox.getTotalMass();

    std::ranges::for_each(
        simBox.getAtoms(),
        [momentumCorrection](auto &atom)
        { atom->addVelocity(-momentumCorrection); }
    );
}

/**
 * @brief reset the angular momentum of the system
 *
 * @details subtract angular momentum correction from all velocities -
 * correction is the total angular momentum divided by the total mass
 *
 * @param simBox
 */
void ResetKinetics::resetAngularMomentum(
    SimulationBox &simBox,
    const Vec3D   &angularMomentum
)
{
    simBox.calculateCenterOfMass();
    const auto centerOfMass = simBox.getCenterOfMass();

    StaticMatrix3x3 helperMatrix{0.0};

    auto addInertiaOfAtom = [&helperMatrix, &centerOfMass](const auto &atom)
    {
        auto       relativePosition = atom->getPosition() - centerOfMass;
        const auto tensor  = tensorProduct(relativePosition, relativePosition);
        helperMatrix      += tensor * atom->getMass();
    };

    std::ranges::for_each(simBox.getAtoms(), addInertiaOfAtom);

    const auto inertia = -helperMatrix + diagonalMatrix(trace(helperMatrix));
    const auto inverseInertia  = inverse(inertia);
    const auto angularVelocity = inverseInertia * angularMomentum;

    auto correctVelocities = [&angularVelocity, &centerOfMass](auto &atom)
    {
        auto relativePosition = atom->getPosition() - centerOfMass;
        atom->addVelocity(-cross(angularVelocity, relativePosition));
    };

    std::ranges::for_each(simBox.getAtoms(), correctVelocities);
}

/**
 * @brief reset the force of the system
 *
 * @details subtract force correction from all forces - correction is the
 * total force divided by the number of atoms
 *
 * @param step
 * @param simBox
 */
void ResetKinetics::resetForces(size_t step, SimulationBox &simBox) const
{
    if (0 != step % _nStepsForcesReset)
        return;

    const auto forceVector     = simBox.calculateTotalForceVector();
    const auto forceCorrection = forceVector / simBox.getNumberOfAtoms();

    std::ranges::for_each(
        simBox.getAtoms(),
        [forceCorrection](auto &atom) { atom->addForce(-forceCorrection); }
    );
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

/**
 * @brief get the number of steps for force reset
 *
 * @return size_t
 */
size_t ResetKinetics::getNStepsForcesReset() const
{
    return _nStepsForcesReset;
}
