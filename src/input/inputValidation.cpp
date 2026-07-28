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

#include "inputFileReader.hpp"

#include <format>   // for format

#include "exceptions.hpp"            // for InputFileException
#include "hessianSettings.hpp"       // for HessianSettings
#include "manostatSettings.hpp"      // for ManostatSettings
#include "potentialSettings.hpp"     // for PotentialSettings
#include "qmSettings.hpp"            // for QMSettings
#include "settings.hpp"              // for Settings
#include "thermostatSettings.hpp"    // for ThermostatSettings
#include "timingsSettings.hpp"       // for TimingsSettings

using namespace input;
using namespace settings;
using namespace customException;

/**
 * @brief validates semantic dependencies between parsed input keywords
 *
 * @details this is intended for cross-keyword checks that cannot be
 * verified within a single keyword parser.
 */
void InputFileReader::validateInputConfiguration() const
{
    validateTimings();
    validateQM();
    validateThermostat();
    validateManostat();
    validateReactionFieldCoulomb();
    validateRingPolymer();
}

/**
 * @brief validates conditionally required timing keywords
 *
 * @throws UserInputException if `nstep` or `timestep` is missing for a job
 * type that requires it
 */
void InputFileReader::validateTimings() const
{
    using enum JobType;

    const auto jobType = Settings::getJobtype();
    const auto requiresNumberOfSteps =
        Settings::isMDJobType() || Settings::isOptJobType() ||
        (jobType == MM_HESSIAN && HessianSettings::optimizeBeforeHessian());

    if (requiresNumberOfSteps && !getKeywordSet("nstep"))
        throw UserInputException(std::format(
            "Job type {} selected. Please set nstep in the input file.",
            string(jobType)
        ));

    if (Settings::isMDJobType() && !getKeywordSet("timestep"))
        throw UserInputException(std::format(
            "Molecular Dynamics job type {} selected. Please set the "
            "time step in the input file.",
            string(jobType)
        ));
}

/**
 * @brief validates QM keyword dependencies
 *
 * @throws InputFileException if selected QM settings require missing or
 * incompatible keywords
 */
void InputFileReader::validateQM() const
{
    if (!Settings::isQMActivated())
        return;

    const auto qmMethod = QMSettings::getQMMethod();

    if (qmMethod == QMMethod::ASEDFTBPLUS)
    {
        auto useThirdOrder = QMSettings::useThirdOrderDftb();

        if (QMSettings::getSlakosType() == SlakosType::THREEOB &&
            !getKeywordSet("third_order"))
            useThirdOrder = true;

        if (!useThirdOrder && getKeywordSet("hubbard_derivs"))
            throw InputFileException(
                "You have set custom Hubbard derivatives but disabled 3rd "
                "order DFTB. This setup is invalid."
            );
    }

    if (qmMethod == QMMethod::FENNOL && !getKeywordSet("fennol_model_path"))
        throw InputFileException(
            "The FeNNol QM runner has been selected but the "
            "\"fennol_model_path\" keyword has not been set. This setup is "
            "invalid."
        );

    if (qmMethod != QMMethod::MACE)
        return;

    const auto modelType    = QMSettings::getMaceModelType();
    const auto model        = QMSettings::getMaceModel();
    const auto modelPathSet = getKeywordSet("mace_model_path");

    if (modelType != MaceModelType::MACE_MP && model != MaceModel::SMALL &&
        model != MaceModel::MEDIUM && model != MaceModel::LARGE)
        throw InputFileException(std::format(
            "The '{}' model size is only compatible with the '{}' model type.",
            string(model),
            string(MaceModelType::MACE_MP)
        ));

    if (model == MaceModel::CUSTOM && !modelPathSet)
        throw InputFileException(
            "You have requested a custom MACE model but haven't provided a "
            "MACE model path."
            "This setup is invalid."
        );

    if (model != MaceModel::CUSTOM && modelPathSet)
        throw InputFileException(
            "You have set a custom MACE model path without requesting a custom "
            "mace model size."
            "This setup is invalid."
        );
}

/**
 * @brief validates thermostat keyword dependencies
 *
 * @throws InputFileException if temperature keywords are missing,
 * contradictory, or define an invalid ramp
 */
void InputFileReader::validateThermostat() const
{
    const auto thermostatType = ThermostatSettings::getThermostatType();

    if (thermostatType != ThermostatType::NONE)
    {
        const auto targetTempDefined = getKeywordSet("temp");
        const auto endTempDefined    = getKeywordSet("end_temp");

        if (!targetTempDefined && !endTempDefined)
            throw InputFileException(std::format(
                "Target or end temperature not set for {} thermostat",
                string(thermostatType)
            ));

        if (targetTempDefined && endTempDefined)
            throw InputFileException(std::format(
                "Both target and end temperature set for {} thermostat. They "
                "are mutually exclusive as they are treated as synonyms",
                string(thermostatType)
            ));
    }

    if (!getKeywordSet("start_temp"))
        return;

    const auto totalSteps = TimingsSettings::getNumberOfSteps();
    const auto rampSteps  = ThermostatSettings::getTemperatureRampSteps();

    if (rampSteps > totalSteps)
        throw InputFileException(std::format(
            "Number of total simulation steps {} is smaller than the "
            "number of temperature ramping steps {}",
            totalSteps,
            rampSteps
        ));

    const auto effectiveRampSteps = rampSteps == 0 ? totalSteps : rampSteps;
    const auto frequency =
        ThermostatSettings::getTemperatureRampFrequency();

    if (frequency > effectiveRampSteps)
        throw InputFileException(std::format(
            "Temperature ramp frequency {} is larger than the number of "
            "ramping steps {}",
            frequency,
            effectiveRampSteps
        ));
}

/**
 * @brief validates manostat keyword dependencies
 *
 * @throws InputFileException if a manostat is selected without `pressure`
 */
void InputFileReader::validateManostat() const
{
    const auto manostatType = ManostatSettings::getManostatType();

    if (manostatType != ManostatType::NONE && !getKeywordSet("pressure"))
        throw InputFileException(std::format(
            "Pressure not set for {} manostat",
            string(manostatType)
        ));
}

/**
 * @brief validates cross-keyword dependencies for the reaction field long range
 * coulomb correction
 *
 * @throws InputFileException if reaction-field Coulomb long-range correction
 * is selected but `rf_epsilon` is missing in the current input file
 */
void InputFileReader::validateReactionFieldCoulomb() const
{
    using enum CoulombLongRangeType;

    const auto longRangeCorrection =
        PotentialSettings::getCoulombLongRangeType();

    if (longRangeCorrection == REACTION_FIELD && !getKeywordSet("rf_epsilon"))
        throw InputFileException(
            "Missing required keyword \"rf_epsilon\" in input file: it must "
            "be set when the Coulomb long-range correction is set to "
            "\"reaction-field\"."
        );
}

/**
 * @brief validates ring-polymer keyword dependencies
 *
 * @throws InputFileException if a ring-polymer job omits `rpmd_n_replica`
 */
void InputFileReader::validateRingPolymer() const
{
    if (Settings::isRingPolymerMDActivated() &&
        !getKeywordSet("rpmd_n_replica"))
        throw InputFileException(
            "Number of beads not set for ring polymer simulation"
        );
}
