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
#include "exceptions.hpp"                    // for exc::UserInputException
#include "globalTimer.hpp"
#include "mathUtilities.hpp"   // for isZero
#include "physicalData.hpp"    // for physicalData::PhysicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "staticMatrix.hpp"    // for operator*, operator+=
#include "staticMatrix/staticMatrix3x3Class.hpp"
#include "thermostatSettings.hpp"   // for settings::ThermostatSettings
#include "vector3d.hpp"             // for Vec3D, Vector3D, cross

namespace resetKinetics
{

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
     * @param step The current simulation step
     * @param physicalData The physical data of the system
     * @param simulationBox The simulation box containing the system
     */
    void ResetKinetics::reset(
        size_t                      step,
        physicalData::PhysicalData &physicalData,
        molsys::SimulationBox      &simulationBox
    )
    {
        auto _ = scopedTimer(TimerId::ResetKinetics, "Reset Kinetics");

        _momentum        = physicalData.getMomentum() * S_TO_FS;
        _angularMomentum = physicalData.getAngularMomentum() * S_TO_FS;
        _temperature     = physicalData.getTemperature();

        auto resetTemp = (step <= _nStepsTemperatureReset);
        resetTemp      = resetTemp || (0 == step % _frequencyTemperatureReset);

        auto resetMom = (step <= _nStepsMomentumReset);
        resetMom      = resetMom || (0 == step % _frequencyMomentumReset);
        resetMom      = !resetTemp && resetMom;

        auto resetAngular = (step <= _nStepsAngularReset);
        resetAngular = resetAngular || (0 == step % _frequencyAngularReset);

        if (resetTemp)
        {
            ResetKinetics::resetTemperature(simulationBox);
            ResetKinetics::resetMomentum(simulationBox);
        }

        if (resetMom)
            ResetKinetics::resetMomentum(simulationBox);

        if (resetAngular)
            ResetKinetics::resetAngularMomentum(simulationBox);

        physicalData.setTemperature(_temperature);
        physicalData.setMomentum(_momentum * FS_TO_S);
        physicalData.setAngularMomentum(_angularMomentum * FS_TO_S);
    }

    /**
     * @brief reset the temperature of the system - hard scaling
     *
     * @details calculate hard scaling factor for target temperature and current
     * temperature and scale all velocities
     *
     * @param simulationBox The simulation box containing the system
     */
    void ResetKinetics::resetTemperature(molsys::SimulationBox &simulationBox)
    {
        const auto targetTemp =
            settings::ThermostatSettings::getActualTargetTemperature();

        if (utilities::isZero(_temperature))
        {
            throw exc::UserInputException(
                "Cannot rescale a zero-temperature system. Initialize "
                "velocities "
                "first."
            );
        }

        const auto lambda = ::sqrt(targetTemp / _temperature);

        std::ranges::for_each(
            simulationBox.getAtoms(),
            [lambda](auto &atom) { atom->scaleVelocity(lambda); }
        );

        _temperature     = simulationBox.calculateTemperature();
        _momentum        = simulationBox.calculateMomentum();
        _angularMomentum = simulationBox.calculateAngularMomentum(_momentum);
    }

    /**
     * @brief reset the momentum of the system
     *
     * @details subtract momentum correction from all velocities - correction is
     * the total momentum divided by the total mass
     *
     * @param simulationBox The simulation box containing the system
     */
    void ResetKinetics::resetMomentum(molsys::SimulationBox &simulationBox)
    {
        const auto momentumVector = _momentum;
        const auto momentumCorrection =
            momentumVector / simulationBox.getTotalMass();

        std::ranges::for_each(
            simulationBox.getAtoms(),
            [momentumCorrection](auto &atom)
            { atom->addVelocity(-momentumCorrection); }
        );

        _temperature     = simulationBox.calculateTemperature();
        _momentum        = simulationBox.calculateMomentum();
        _angularMomentum = simulationBox.calculateAngularMomentum(_momentum);
    }

    /**
     * @brief reset the angular momentum of the system
     *
     * @details subtract angular momentum correction from all velocities -
     * correction is the total angular momentum divided by the total mass
     *
     * @param simulationBox The simulation box containing the system
     */
    void ResetKinetics::resetAngularMomentum(
        molsys::SimulationBox &simulationBox
    )
    {
        simulationBox.calculateCenterOfMass();
        const auto centerOfMass = simulationBox.getCenterOfMass();

        _angularMomentum = simulationBox.calculateAngularMomentum(_momentum);

        linalg::tensor3D helperMatrix{0.0};

        auto addInertiaOfAtom = [&helperMatrix, &centerOfMass](const auto &atom)
        {
            auto       relativePosition = atom->getPosition() - centerOfMass;
            const auto tensor =
                tensorProduct(relativePosition, relativePosition);
            helperMatrix += tensor * atom->getMass();
        };

        std::ranges::for_each(simulationBox.getAtoms(), addInertiaOfAtom);

        const auto inertia =
            -helperMatrix + linalg::diagonalMatrix(trace(helperMatrix));
        const auto inverseInertia  = inverse(inertia);
        const auto angularVelocity = inverseInertia * _angularMomentum;

        auto correctVelocities = [&angularVelocity, &centerOfMass](auto &atom)
        {
            auto relativePosition = atom->getPosition() - centerOfMass;
            atom->addVelocity(-cross(angularVelocity, relativePosition));
        };

        std::ranges::for_each(simulationBox.getAtoms(), correctVelocities);

        _temperature     = simulationBox.calculateTemperature();
        _momentum        = simulationBox.calculateMomentum();
        _angularMomentum = simulationBox.calculateAngularMomentum(_momentum);
    }

    /**
     * @brief reset the force of the system
     *
     * @details subtract force correction from all forces - correction is the
     * total force divided by the number of atoms
     *
     * @param step
     * @param simulationBox The simulation box containing the system
     */
    void ResetKinetics::resetForces(
        size_t                 step,
        molsys::SimulationBox &simulationBox
    ) const
    {
        if (0 != step % _nStepsForcesReset)
            return;

        const auto forceVector = simulationBox.calculateTotalForceVector();
        const auto forceCorrection =
            forceVector / simulationBox.getNumberOfAtoms();

        std::ranges::for_each(
            simulationBox.getAtoms(),
            [forceCorrection](auto &atom) { atom->addForce(-forceCorrection); }
        );
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
    void ResetKinetics::setTemperature(double temperature)
    {
        _temperature = temperature;
    }

    /**
     * @brief set the momentum
     *
     * @param momentum
     */
    void ResetKinetics::setMomentum(const linalg::Vec3D &momentum)
    {
        _momentum = momentum;
    }

    /**
     * @brief set the angular momentum
     *
     * @param angularMomentum
     */
    void ResetKinetics::setAngularMomentum(const linalg::Vec3D &angularMomentum)
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

    /**
     * @brief get the number of steps for force reset
     *
     * @return size_t
     */
    size_t ResetKinetics::getNStepsForcesReset() const
    {
        return _nStepsForcesReset;
    }

}   // namespace resetKinetics
