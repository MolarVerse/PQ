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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <optional>
#include <string>

#include "constants/conversionFactors.hpp"
#include "constants/internalConversionFactors.hpp"
#include "exceptions.hpp"
#include "globalTimer.hpp"
#include "mathUtilities.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "staticMatrix.hpp"
#include "staticMatrix/staticMatrix3x3Class.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace kinetics
{
    static constexpr double FRACTION_TO_PERCENT = 100.0;

    /**
     * @brief Construct a new Reset Kinetics:: Reset Kinetics object
     *
     * @param settings The settings for resetting kinetics
     */
    ResetKinetics::ResetKinetics(const ResetKineticsSettings &settings)
        : _settings(settings)
    {
    }

    /**
     * @brief checks to reset angular momentum
     *
     * @param step The current simulation step
     * @param physicalData The physical data of the system
     * @param simulationBox The simulation box containing the system
     * @param targetTemperature The target temperature for the system
     */
    void ResetKinetics::reset(
        size_t                      step,
        physicalData::PhysicalData &physicalData,
        molsys::SimulationBox      &simulationBox,
        const Temperature          &targetTemperature
    ) const
    {
        auto _ = scopedTimer(TimerId::ResetKinetics, "Reset Kinetics");

        auto momentum        = physicalData.getMomentum() * S_TO_FS;
        auto angularMomentum = physicalData.getAngularMomentum() * S_TO_FS;
        auto temperature     = physicalData.getTemperature();

        auto resetTemp = (step <= _settings->getNScale());
        resetTemp      = resetTemp || (0 == step % _settings->getFScale());

        auto resetMom = (step <= _settings->getNReset());
        resetMom      = resetMom || (0 == step % _settings->getFReset());

        auto resetAngular = (step <= _settings->getNResetAngular());
        resetAngular =
            resetAngular || (0 == step % _settings->getFResetAngular());

        if (resetTemp)
        {
            resetTemperature(simulationBox, temperature, targetTemperature);
            momentum = simulationBox.calculateMomentum();
            resetMomentum(simulationBox, momentum);
            temperature     = simulationBox.calculateTemperature();
            momentum        = simulationBox.calculateMomentum();
            angularMomentum = simulationBox.calculateAngularMomentum(momentum);
        }
        else if (resetMom)
        {
            // temperature also needs reset of momentum, thus the else if
            ResetKinetics::resetMomentum(simulationBox, momentum);
            momentum        = simulationBox.calculateMomentum();
            temperature     = simulationBox.calculateTemperature();
            angularMomentum = simulationBox.calculateAngularMomentum(momentum);
        }

        if (resetAngular)
        {
            ResetKinetics::resetAngularMomentum(simulationBox, angularMomentum);
            temperature     = simulationBox.calculateTemperature();
            momentum        = simulationBox.calculateMomentum();
            angularMomentum = simulationBox.calculateAngularMomentum(momentum);
        }

        physicalData.setTemperature(temperature);
        physicalData.setMomentum(momentum * FS_TO_S);
        physicalData.setAngularMomentum(angularMomentum * FS_TO_S);
    }

    /**
     * @brief reset the temperature of the system - hard scaling
     *
     * @details calculate hard scaling factor for target temperature and current
     * temperature and scale all velocities
     *
     * @param simulationBox The simulation box containing the system
     * @param temperature The current temperature of the system
     * @param targetTemperature The target temperature to which the system
     * should be scaled
     *
     */
    void ResetKinetics::resetTemperature(
        molsys::SimulationBox &simulationBox,
        const Temperature     &temperature,
        const Temperature     &targetTemperature
    )
    {
        const auto targetTemp = targetTemperature;

        if (utilities::isZero(temperature))
        {
            throw exc::UserInputException(
                "Cannot rescale a zero-temperature system. Initialize "
                "velocities "
                "first."
            );
        }

        const auto lambda = mstd::sqrt(tempRatio(targetTemp, temperature));

        std::ranges::for_each(
            simulationBox.getAtoms(),
            [lambda](auto &atom) { atom->scaleVelocity(lambda); }
        );
    }

    /**
     * @brief reset the momentum of the system
     *
     * @details subtract momentum correction from all velocities - correction is
     * the total momentum divided by the total mass
     *
     * @param simulationBox The simulation box containing the system
     * @param momentum the current momentum of the system
     */
    void ResetKinetics::resetMomentum(
        molsys::SimulationBox &simulationBox,
        const linalg::Vec3D   &momentum
    )
    {
        const auto momentumCorrection = momentum / simulationBox.getTotalMass();

        std::ranges::for_each(
            simulationBox.getAtoms(),
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
     * @param simulationBox The simulation box containing the system
     * @param angularMomentum the current angular momentum of the system
     */
    void ResetKinetics::resetAngularMomentum(
        molsys::SimulationBox &simulationBox,
        const linalg::Vec3D   &angularMomentum
    )
    {
        simulationBox.calculateCenterOfMass();
        const auto centerOfMass = simulationBox.getCenterOfMass();

        linalg::StaticMatrix3x3 helperMatrix{0.0};

        auto addInertiaOfAtom = [&helperMatrix, &centerOfMass](const auto &atom)
        {
            auto       relativePosition = atom->getPosition() - centerOfMass;
            const auto tensor =
                tensorProduct(relativePosition, relativePosition);
            helperMatrix += tensor * atom->getMass();
        };

        std::ranges::for_each(simulationBox.getAtoms(), addInertiaOfAtom);

        const auto inertia =
            -helperMatrix + linalg::diagonalMatrix(linalg::trace(helperMatrix));
        const auto inverseInertia  = inverse(inertia);
        const auto angularVelocity = inverseInertia * angularMomentum;

        auto correctVelocities = [&angularVelocity, &centerOfMass](auto &atom)
        {
            auto relativePosition = atom->getPosition() - centerOfMass;
            atom->addVelocity(-cross(angularVelocity, relativePosition));
        };

        std::ranges::for_each(simulationBox.getAtoms(), correctVelocities);
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
        if (0 != step % _settings->getFResetForces())
            return;

        const auto forceVector = simulationBox.calculateTotalForceVector();
        const auto forceCorrection =
            forceVector / simulationBox.getNumberOfAtoms();

        std::ranges::for_each(
            simulationBox.getAtoms(),
            [forceCorrection](auto &atom) { atom->addForce(-forceCorrection); }
        );
    }

    /**
     * @brief warn once if the total momentum drifts above the threshold
     *
     * @details warns if the centre-of-mass velocity P/M exceeds
     * momentum_warn_threshold times the thermal velocity sqrt(kT/m) of the
     * mean atomic mass m = M/N, i.e. if P^2 > threshold^2 * N * M * kT
     *
     * @param step The current simulation step
     * @param physicalData The physical data before the kinetics reset
     * @param simulationBox The simulation box containing the system
     * @return the warning message, only for the first drift of a run
     */
    std::optional<std::string> ResetKinetics::checkMomentumDrift(
        size_t                            step,
        const physicalData::PhysicalData &physicalData,
        const molsys::SimulationBox      &simulationBox
    )
    {
        const auto threshold = _settings->getMomentumWarnThreshold();

        if (_momentumDriftWarned || threshold <= 0.0)
        {
            return std::nullopt;
        }

        const auto advice = _momentumDriftAdvice(step);

        if (!advice)
        {
            return std::nullopt;
        }

        const auto momentum    = physicalData.getMomentum();
        const auto temperature = physicalData.getTemperature()
                                     .quantity_from_zero()
                                     .numerical_value_in(K);
        const auto nAtoms =
            static_cast<double>(simulationBox.getNumberOfAtoms());

        // momentum in amu*Angstrom/s, the velocity unit of TEMPERATURE_FACTOR
        const auto momentumSquared        = normSquared(momentum * S_TO_FS);
        const auto thermalMomentumSquared = temperature * nAtoms *
                                            simulationBox.getTotalMass() /
                                            TEMPERATURE_FACTOR;

        if (momentumSquared <= threshold * threshold * thermalMomentumSquared)
        {
            return std::nullopt;
        }

        _momentumDriftWarned = true;

        return std::format(
            "Total momentum reached {:.3e} amu*Angstrom/fs at step {}: the "
            "centre of mass moves faster than {:g} % of the thermal velocity "
            "{}. Set momentum_warn_threshold = 0 to disable this warning.",
            norm(momentum),
            step,
            threshold * FRACTION_TO_PERCENT,
            *advice
        );
    }

    /**
     * @brief explain why the momentum can drift with the current reset settings
     *
     * @param step The current simulation step
     * @return the advice, std::nullopt if the momentum is reset in this step
     */
    std::optional<std::string> ResetKinetics::_momentumDriftAdvice(
        size_t step
    ) const
    {
        const auto initialSteps = _settings->getInitialMomentumResetSteps();
        const auto interval     = _settings->getMomentumResetInterval();

        if (step <= initialSteps)
        {
            return std::nullopt;
        }

        if (interval)
        {
            if (*interval <= 1)
            {
                return std::nullopt;
            }

            return std::format(
                "between the momentum resets every {} steps (freset/fscale). "
                "Set a smaller freset to remove it more often",
                *interval
            );
        }

        if (initialSteps > 0)
        {
            return std::format(
                "and the momentum is only removed in the first {} steps "
                "(nreset/nscale). Set freset (e.g. freset = 100) to remove it "
                "periodically",
                initialSteps
            );
        }

        return "and no momentum reset is set. Set freset (e.g. freset = 100) "
               "to remove it periodically";
    }

}   // namespace kinetics
