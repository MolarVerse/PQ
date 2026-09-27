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

#include <algorithm>   // for max
#include <cmath>       // for isfinite
#include <format>      // for format

#include "constants/conversionFactors.hpp"
#include "exceptions.hpp"        // for exc::InputFileException
#include "hessianSettings.hpp"   // for settings::HessianSettings
#include "inputFileReader.hpp"
#include "manostatSettings.hpp"        // for settings::ManostatSettings
#include "optimizerSettings.hpp"       // for settings::OptimizerSettings
#include "potentialSettings.hpp"       // for settings::PotentialSettings
#include "qmSettings.hpp"              // for settings::QMSettings
#include "settings.hpp"                // for settings::Settings
#include "simulationBoxSettings.hpp"   // for SimulationBoxSettings
#include "thermostatSettings.hpp"      // for settings::ThermostatSettings
#include "timingsSettings.hpp"         // for settings::TimingsSettings

namespace input
{

    namespace
    {
        /**
         * @brief validates settings used by active optimization jobs
         *
         * @throws exc::UserInputException if the learning-rate strategy or
         * bounds are invalid
         */
        void validateOptimizer()
        {
            const auto optimizerActive =
                settings::Settings::isOptJobType() ||
                (settings::Settings::getJobtype() ==
                     settings::JobType::MM_HESSIAN &&
                 settings::HessianSettings::optimizeBeforeHessian());

            if (!optimizerActive)
                return;

            settings::OptimizerSettings::validateLearningRateStrategy();
            settings::OptimizerSettings::validateLearningRateBounds();
        }
    }   // namespace

    /**
     * @brief validates semantic dependencies between parsed input keywords
     *
     * @details this is intended for cross-keyword checks that cannot be
     * verified within a single keyword parser.
     */
    void InputFileReader::validateInputConfiguration() const
    {
        _validateTimings();
        validateOptimizer();
        _validateQM();
        _validateThermostat();
        _validateManostat();
        _validateCellList();
        _validateReactionFieldCoulomb();
        _validateRingPolymer();
    }

    /**
     * @brief validates conditionally required timing keywords
     *
     * @throws exc::UserInputException if `nstep` or `timestep` is missing for a
     * job type that requires it
     */
    void InputFileReader::_validateTimings() const
    {
        using enum settings::JobType;

        const auto jobType = settings::Settings::getJobtype();
        const auto requiresNumberOfSteps =
            settings::Settings::isMDJobType() ||
            settings::Settings::isOptJobType() ||
            (jobType == MM_HESSIAN &&
             settings::HessianSettings::optimizeBeforeHessian());

        if (requiresNumberOfSteps && !getKeywordSet("nstep"))
        {
            throw exc::UserInputException(
                std::format(
                    "Job type {} selected. Please set nstep in the input file.",
                    string(jobType)
                )
            );
        }

        if (settings::Settings::isMDJobType() && !getKeywordSet("timestep"))
        {
            throw exc::UserInputException(
                std::format(
                    "Molecular Dynamics job type {} selected. Please set the "
                    "time step in the input file.",
                    string(jobType)
                )
            );
        }
    }

    /**
     * @brief validates QM keyword dependencies
     *
     * @throws exc::InputFileException if selected QM settings require missing
     * or incompatible keywords
     */
    void InputFileReader::_validateQM() const
    {
        if (!settings::Settings::isQMActivated())
            return;

        if (!getKeywordSet("qm_prog"))
            throw exc::InputFileException(
                "QM job selected but the \"qm_prog\" keyword has not been set"
            );

        const auto qmMethod = settings::QMSettings::getQMMethod();

        if (qmMethod == settings::QMMethod::ASEDFTBPLUS)
        {
            if (settings::QMSettings::getSlakosType() ==
                settings::SlakosType::NONE)
                throw exc::InputFileException(
                    "ASE-DFTB+ requires slakos to be 3ob, matsci, or custom"
                );

            if (settings::QMSettings::getSlakosType() ==
                    settings::SlakosType::CUSTOM &&
                !getKeywordSet("slakos_path"))
            {
                throw exc::InputFileException(
                    "Custom Slater-Koster parameters require the "
                    "\"slakos_path\" keyword"
                );
            }

            auto useThirdOrder = settings::QMSettings::useThirdOrderDftb();

            if (settings::QMSettings::getSlakosType() ==
                    settings::SlakosType::THREEOB &&
                !getKeywordSet("third_order"))
                useThirdOrder = true;

            if (!useThirdOrder && getKeywordSet("hubbard_derivs"))
            {
                throw exc::InputFileException(
                    "You have set custom Hubbard derivatives but disabled 3rd "
                    "order DFTB. This setup is invalid."
                );
            }
        }

        if (qmMethod == settings::QMMethod::FENNOL &&
            !getKeywordSet("fennol_model_path"))
        {
            throw exc::InputFileException(
                "The FeNNol QM runner has been selected but the "
                "\"fennol_model_path\" keyword has not been set. This setup is "
                "invalid."
            );
        }

        if (qmMethod != settings::QMMethod::MACE)
            return;

        const auto modelType    = settings::QMSettings::getMaceModelType();
        const auto model        = settings::QMSettings::getMaceModel();
        const auto modelPathSet = getKeywordSet("mace_model_path");

        if (modelType != settings::MaceModelType::MACE_MP &&
            model != settings::MaceModel::SMALL &&
            model != settings::MaceModel::MEDIUM &&
            model != settings::MaceModel::LARGE)
        {
            throw exc::InputFileException(
                std::format(
                    "The '{}' model size is only compatible with the '{}' "
                    "model "
                    "type.",
                    string(model),
                    string(settings::MaceModelType::MACE_MP)
                )
            );
        }

        if (model == settings::MaceModel::CUSTOM && !modelPathSet)
        {
            throw exc::InputFileException(
                "You have requested a custom MACE model but haven't provided a "
                "MACE model path."
                "This setup is invalid."
            );
        }

        if (model != settings::MaceModel::CUSTOM && modelPathSet)
        {
            throw exc::InputFileException(
                "You have set a custom MACE model path without requesting a "
                "custom "
                "mace model size."
                "This setup is invalid."
            );
        }
    }

    /**
     * @brief validates thermostat keyword dependencies
     *
     * @throws exc::InputFileException if temperature keywords are missing,
     * contradictory, or define an invalid ramp
     */
    void InputFileReader::_validateThermostat() const
    {
        const auto thermostatType =
            settings::ThermostatSettings::getThermostatType();
        const auto targetTempDefined = getKeywordSet("temp");
        const auto startTempDefined  = getKeywordSet("start_temp");
        const auto endTempDefined    = getKeywordSet("end_temp");

        if (thermostatType != settings::ThermostatType::NONE)
        {
            if (!targetTempDefined && !endTempDefined)
            {
                throw exc::InputFileException(
                    std::format(
                        "Target or end temperature not set for {} thermostat",
                        string(thermostatType)
                    )
                );
            }

            if (targetTempDefined && endTempDefined)
            {
                throw exc::InputFileException(
                    std::format(
                        "Both target and end temperature set for {} "
                        "thermostat. "
                        "They are mutually exclusive as they are treated as "
                        "synonyms",
                        string(thermostatType)
                    )
                );
            }
        }

        if (settings::SimulationBoxSettings::getInitializeVelocities() !=
                settings::InitVelocities::FALSE &&
            !targetTempDefined && !startTempDefined && !endTempDefined)
            throw exc::InputFileException(
                "Initializing velocities requires temp, start_temp, or end_temp"
            );

        if (settings::Settings::isMDJobType() &&
            (thermostatType == settings::ThermostatType::BERENDSEN ||
             thermostatType == settings::ThermostatType::VELOCITY_RESCALING))
        {
            const auto relaxationTime =
                settings::ThermostatSettings::getRelaxationTime() * PS_TO_FS;

            if (settings::TimingsSettings::getTimeStep() > relaxationTime)
            {
                throw exc::InputFileException(
                    "The timestep must not exceed the thermostat relaxation "
                    "time"
                );
            }
        }

        if (thermostatType == settings::ThermostatType::LANGEVIN)
        {
            auto maxTemperature = 0.0;

            if (targetTempDefined)
            {
                maxTemperature = std::max(
                    maxTemperature,
                    settings::ThermostatSettings::getTargetTemperature()
                );
            }
            if (startTempDefined)
            {
                maxTemperature = std::max(
                    maxTemperature,
                    settings::ThermostatSettings::getStartTemperature()
                );
            }
            if (endTempDefined)
            {
                maxTemperature = std::max(
                    maxTemperature,
                    settings::ThermostatSettings::getEndTemperature()
                );
            }

            const auto unitConversion = M2_TO_ANGSTROM2 * KG_TO_GRAM / FS_TO_S;
            const auto conversionFactor =
                UNIVERSAL_GAS_CONSTANT * unitConversion;
            const auto sigmaSquared =
                4.0 * settings::ThermostatSettings::getFriction() *
                conversionFactor * maxTemperature /
                settings::TimingsSettings::getTimeStep();

            if (!std::isfinite(sigmaSquared))
            {
                throw exc::InputFileException(
                    "Langevin thermostat parameters produce a non-finite "
                    "random-force scale"
                );
            }
        }

        if (thermostatType == settings::ThermostatType::NOSE_HOOVER)
        {
            if (targetTempDefined &&
                settings::ThermostatSettings::getTargetTemperature() <= 0.0)
                throw exc::InputFileException(
                    "Nose-Hoover target temperature must be greater than zero"
                );

            if (endTempDefined &&
                settings::ThermostatSettings::getEndTemperature() <= 0.0)
                throw exc::InputFileException(
                    "Nose-Hoover end temperature must be greater than zero"
                );

            if (startTempDefined &&
                settings::ThermostatSettings::getStartTemperature() <= 0.0)
                throw exc::InputFileException(
                    "Nose-Hoover start temperature must be greater than zero"
                );
        }

        if (!startTempDefined)
            return;

        const auto totalSteps = settings::TimingsSettings::getNumberOfSteps();
        const auto rampSteps =
            settings::ThermostatSettings::getTemperatureRampSteps();

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
        const auto frequency =
            settings::ThermostatSettings::getTemperatureRampFrequency();

        if (frequency > effectiveRampSteps)
        {
            throw exc::InputFileException(
                std::format(
                    "Temperature ramp frequency {} is larger than the number "
                    "of "
                    "ramping steps {}",
                    frequency,
                    effectiveRampSteps
                )
            );
        }
    }

    /**
     * @brief validates manostat keyword dependencies
     *
     * @throws exc::InputFileException if a manostat is selected without
     * `pressure`
     */
    void InputFileReader::_validateManostat() const
    {
        const auto manostatType = settings::ManostatSettings::getManostatType();

        if (manostatType == ManostatType::NONE)
            return;

        if (!getKeywordSet("pressure"))
        {
            throw exc::InputFileException(
                std::format(
                    "Pressure not set for {} manostat",
                    ManostatTypeMeta::toString(manostatType)
                )
            );
        }

        const auto relaxationTime =
            settings::ManostatSettings::getTauManostat() * PS_TO_FS;

        if (settings::TimingsSettings::getTimeStep() > relaxationTime)
            throw exc::InputFileException(
                "The timestep must not exceed the manostat relaxation time"
            );
    }

    /**
     * @brief validates cell-list dependencies
     *
     * @throws exc::InputFileException if an active cell list is incompatible
     * with the selected potential
     */
    void InputFileReader::_validateCellList()
    {
        if (!settings::Settings::isCellListActivated())
            return;

        if (settings::Settings::isQMOnlyActivated())
            throw exc::InputFileException(
                "Cell lists are not available for pure QM simulations"
            );

        if (settings::PotentialSettings::getCoulombRadiusCutOff() <= 0.0)
            throw exc::InputFileException(
                "An active cell list requires rcoulomb to be greater than zero"
            );
    }

    /**
     * @brief validates cross-keyword dependencies for the reaction field long
     * range coulomb correction
     *
     * @throws exc::InputFileException if reaction-field Coulomb long-range
     * correction is selected but `rf_epsilon` is missing in the current input
     * file
     */
    void InputFileReader::_validateReactionFieldCoulomb() const
    {
        using enum settings::CoulombLongRangeType;

        const auto longRangeCorrection =
            settings::PotentialSettings::getCoulombLongRangeType();

        if (longRangeCorrection == REACTION_FIELD &&
            !getKeywordSet("rf_epsilon"))
        {
            throw exc::InputFileException(
                "Missing required keyword \"rf_epsilon\" in input file: it "
                "must "
                "be set when the Coulomb long-range correction is set to "
                "\"reaction-field\"."
            );
        }
    }

    /**
     * @brief validates ring-polymer keyword dependencies
     *
     * @throws exc::InputFileException if a ring-polymer job omits
     * `rpmd_n_replica`
     */
    void InputFileReader::_validateRingPolymer() const
    {
        if (settings::Settings::isRingPolymerMDActivated() &&
            !getKeywordSet("rpmd_n_replica"))
            throw exc::InputFileException(
                "Number of beads not set for ring polymer simulation"
            );
    }

}   // namespace input
