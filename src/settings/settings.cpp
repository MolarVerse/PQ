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

#include "settings.hpp"

#include <cmath>
#include <format>

#include "constants/conversionFactors.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "hybridSettings.hpp"
#include "resetKineticsSettings.hpp"
#include "timingsSettings.hpp"
#include "unit_types.hpp"
#include "units.hpp"

namespace
{
    /**
     * @brief validates the resolved thermostat and temperature ramp values
     *
     * @details requires Settings::finalize() to have been called, as the end
     * temperature and ramp parameters are resolved there.
     *
     * @throws exc::InputFileException if the values are inconsistent
     */
    void validateThermostat(
        const ThermostatSettings& thermostatSettings,
        const KineticsSettings&   kinetics
    )
    {
        const auto thermostatType = thermostatSettings.getThermostatType();
        const auto hasRamp        = kinetics.isStartTemperatureSet();

        if (settings::GeneralSettings::isMDJobType() &&
            (thermostatType == ThermostatType::BERENDSEN ||
             thermostatType == ThermostatType::VELOCITY_RESCALING))
        {
            if (settings::TimingsSettings::getTimeStep() >
                ThermostatSettings::getRelaxationTime())
            {
                throw exc::InputFileException(
                    "The timestep must not exceed the thermostat relaxation "
                    "time"
                );
            }
        }

        if (thermostatType == ThermostatType::LANGEVIN)
        {
            auto maxTemperature = kinetics.getEndTemperature();

            if (hasRamp)
            {
                maxTemperature =
                    mstd::max(maxTemperature, kinetics.getStartTemperature());
            }

            const auto unitConversion = M2_TO_ANGSTROM2 * KG_TO_GRAM / FS_TO_S;
            const auto conversionFactor =
                UNIVERSAL_GAS_CONSTANT * unitConversion;
            const auto sigmaSquared =
                4.0 * ThermostatSettings::getFriction() * conversionFactor *
                maxTemperature.quantity_from_zero().numerical_value_in(K) /
                settings::TimingsSettings::getTimeStep().numerical_value_in(fs);

            if (!std::isfinite(sigmaSquared))
            {
                throw exc::InputFileException(
                    "Langevin thermostat parameters produce a non-finite "
                    "random-force scale"
                );
            }
        }

        if (thermostatType == ThermostatType::NOSE_HOOVER)
        {
            if (kinetics.getEndTemperature() <= Temperature::zero())
                throw exc::InputFileException(
                    "Nose-Hoover target temperature must be greater than zero"
                );

            if (hasRamp &&
                kinetics.getStartTemperature() <= Temperature::zero())
                throw exc::InputFileException(
                    "Nose-Hoover start temperature must be greater than zero"
                );
        }

        if (!hasRamp)
            return;

        const auto totalSteps = settings::TimingsSettings::getNumberOfSteps();
        const auto rampSteps  = kinetics.getRampingSteps();

        if (rampSteps > totalSteps)
        {
            throw exc::InputFileException(
                std::format(
                    "Number of total simulation steps {} is smaller than the "
                    "number of temperature ramping steps {}",
                    totalSteps,
                    rampSteps
                )
            );
        }

        const auto effectiveRampSteps = rampSteps == 0 ? totalSteps : rampSteps;
        const auto frequency          = kinetics.getRampingFrequency();

        if (frequency > effectiveRampSteps)
        {
            throw exc::InputFileException(
                std::format(
                    "Temperature ramp frequency {} is larger than the number "
                    "of ramping steps {}",
                    frequency,
                    effectiveRampSteps
                )
            );
        }
    }
}   // namespace

/**
 * @brief Finalize the settings by finalizing each individual setting component
 */
void Settings::finalize()
{
    resetKinetics.finalize();
    kinetics.finalize();
    manostatSettings.finalize(referenceSettings);
    outputFiles.finalize();
    waterModelSettings.finalize(referenceSettings);
    constraintSettings.finalize(referenceSettings, waterModelSettings);
    thermostatSettings.finalize(referenceSettings);
    settings::HybridSettings::finalize(fileSettings);

    const auto& qmMethod = qmSettings.getQMMethod();

    switch (qmMethod)
    {
        case QMMethod::ASE_DFTBPLUS:
            dftbSettings.finalize(referenceSettings);
            break;
        case QMMethod::ASE_XTB: xtbSettings.finalize(referenceSettings); break;
        case QMMethod::MACE:
        case QMMethod::DFTBPLUS:
        case QMMethod::FENNOL:
        case QMMethod::PYSCF:
        case QMMethod::TURBOMOLE:
        case QMMethod::ORCA:
        case QMMethod::NONE: break;
    }

    if (qmSettings.isExternalQMRunner())
        externalQMSettings.finalize();
}

/**
 * @brief Validate the finalized settings
 *
 * @details checks that depend on values resolved in finalize(), so this must
 * be called after finalize()
 */
void Settings::validate() const
{
    validateThermostat(thermostatSettings, kinetics);
}
