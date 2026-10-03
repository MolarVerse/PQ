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

#include "thermostatInputParser.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

#include "constants.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "rangeValidator.hpp"
#include "references.hpp"
#include "referencesOutput.hpp"
#include "thermostatSettings.hpp"

namespace input
{

    /**
     * @brief Construct a new Input File Parser Thermostat:: Input File Parser
     * Thermostat object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) thermostat "<string>" 2)
     * temp
     * "<double>" 3) t_relaxation "<double>" 4) friction "<double>" 5)
     * nh-chain_length "<size_t>" 6) coupling_frequency "<double>"
     */
    ThermostatInputParser::ThermostatInputParser()
    {
        addThermostatKey();
        addTemperatureKey();
        addStartTemperatureKey();
        addEndTemperatureKey();
        addTemperatureRampStepsKey();
        addTemperatureRampFrequencyKey();
        addThermostatRelaxationTimeKey();
        addThermostatFrictionKey();
        addThermostatChainLengthKey();
        addThermostatCouplingFrequencyKey();
    }

    void ThermostatInputParser::addThermostatKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "thermostat",
            .title = "Thermostat Type",
            .description =
                "Specifies the type of thermostat used in the simulation"
        };

        const auto setValue = [&](ThermostatType value)
        {
            settings::ThermostatSettings::setThermostatType(value);
            switch (value)
            {
                case ThermostatType::BERENDSEN:
                    references::ReferencesOutput::addReferenceFile(
                        references::BERENDSEN_FILE
                    );
                    break;
                case ThermostatType::VELOCITY_RESCALING:
                    references::ReferencesOutput::addReferenceFile(
                        references::VELOCITY_RESCALING_FILE
                    );
                    break;
                case ThermostatType::LANGEVIN:
                    references::ReferencesOutput::addReferenceFile(
                        references::LANGEVIN_FILE
                    );
                    break;
                case ThermostatType::NOSE_HOOVER:
                    references::ReferencesOutput::addReferenceFile(
                        references::NOSE_HOOVER_CHAIN_FILE
                    );
                    break;
                case ThermostatType::NONE: break;
            }
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<ThermostatType>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addTemperatureKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "temp",
            .title       = "Target Temperature",
            .description = "Specifies the target temperature for the simulation"
        };

        const auto setValue = [&](double value)
        { settings::ThermostatSettings::setTargetTemperature(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTEZeroDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addStartTemperatureKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "start_temp",
            .title       = "Start Temperature",
            .description = "Specifies the start temperature for the simulation"
        };

        const auto setValue = [&](double value)
        { settings::ThermostatSettings::setStartTemperature(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTEZeroDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addEndTemperatureKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "end_temperature",
            .title       = "End Temperature",
            .description = "Specifies the end temperature for the simulation"
        };

        const auto setValue = [&](double value)
        { settings::ThermostatSettings::setEndTemperature(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTEZeroDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addTemperatureRampStepsKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "temp_ramp_steps",
            .title = "Temperature Ramp Steps",
            .description =
                "Specifies the number of steps for the temperature ramp"
        };

        const auto setValue = [&](size_t value)
        { settings::ThermostatSettings::setTemperatureRampSteps(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<size_t>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addTemperatureRampFrequencyKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "temp_ramp_frequency",
            .title       = "Temperature Ramp Frequency",
            .description = "Specifies the frequency of the temperature ramp"
        };

        const auto setValue = [&](size_t value)
        { settings::ThermostatSettings::setTemperatureRampFrequency(value); };

        const auto validator =
            RangeValidator<size_t, Greater::GT>{0, std::nullopt};

        auto &key = _getRegistry().registerKey(
            KeyRegistry<size_t>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(validator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addThermostatRelaxationTimeKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "t_relaxation",
            .title       = "Relaxation Time",
            .description = "Specifies the relaxation time of the thermostat"
        };

        const auto setValue = [&](double value)
        { settings::ThermostatSettings::setRelaxationTime(value); };

        const auto validator = RangeValidator<double, Greater::GT, Less::LE>{
            0.0,
            std::numeric_limits<double>::max() / PS_TO_FS
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(validator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addThermostatFrictionKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "friction",
            .title       = "Friction",
            .description = "Specifies the friction of the thermostat"
        };

        const auto setValue = [&](double value)
        {
            settings::ThermostatSettings::setFriction(
                value * NOSE_HOVER_FRICTION_INPUT_TO_INTERNAL
            );
        };

        const auto validator = RangeValidator<double, Greater::GE, Less::LE>{
            0.0,
            std::numeric_limits<double>::max() /
                NOSE_HOVER_FRICTION_INPUT_TO_INTERNAL
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(validator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addThermostatChainLengthKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "nh_chain_length",
            .title = "Thermostat Chain Length",
            .description =
                "Specifies the chain length of the nh-chain thermostat"
        };

        const auto setValue = [&](size_t value)
        { settings::ThermostatSettings::setNoseHooverChainLength(value); };

        const auto minValidator =
            RangeValidator<size_t, Greater::GE>{1, std::nullopt};

        auto &key = _getRegistry().registerKey(
            KeyRegistry<size_t>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(minValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ThermostatInputParser::addThermostatCouplingFrequencyKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "coupling_frequency",
            .title = "Thermostat Coupling Frequency",
            .description =
                "Specifies the coupling frequency of the nh-chain thermostat"
        };

        const auto setValue = [&](double value)
        {
            settings::ThermostatSettings::setNoseHooverCouplingFrequency(value);
        };

        const auto validator = RangeValidator<double, Greater::GE, Less::LE>{
            0.0,
            std::sqrt(std::numeric_limits<double>::max()) / PER_CM_TO_HZ
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(validator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

}   // namespace input
