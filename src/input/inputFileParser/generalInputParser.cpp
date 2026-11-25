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

#include "generalInputParser.hpp"

#include <cstdint>
#include <optional>

#include "enums/general.hpp"
#include "generalSettings.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "keyValidatorBase.hpp"
#include "rangeValidator.hpp"
#include "stringUtilities.hpp"

namespace input
{

    /**
     * @brief Construct a new Input File Parser General:: Input File Parser
     * General object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) jobtype "<string>"
     * (required)
     *
     */
    GeneralInputParser::GeneralInputParser(GeneralSettings &generalSettings)
        : _generalSettings(generalSettings)
    {
        addJobTypeKey();
        addDimensionalityKey();
        addFloatingPointTypeKey();
        addRandomSeedKey();
    }

    /**
     * @brief Adds the jobtype key to the input parser and validates its value
     */
    void GeneralInputParser::addJobTypeKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "jobtype",
            .title       = "Job Type",
            .description = "Specifies the type of job to be performed",
            .errorMessage =
                "Invalid job type specified - possible values "
                "are:\n"
                "- mm-opt\n"
                "- mm-hessian\n"
                "- mm-md\n"
                "- qm-md\n"
                "- qm-rpmd\n"
                "- qmmm-md\n"
        };

        const auto setValue = [this](JobType jobType)
        { _generalSettings->setJobtypeByUser(jobType); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<JobType>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), true);
    }

    void GeneralInputParser::addDimensionalityKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "dim",
            .title = "Dimensionality",
            .description =
                "Specifies the dimensionality of the simulation (3 or 3d)",
            .errorMessage = "Dimensionality must be 3 or 3d",
            .lifecycle    = deprecate(
                {.deprecatedSince = v1_1_0,
                    .removedIn       = v2_0_0,
                    .message = "Since this key is only possible to be used for "
                                  "3-dimensional simulations, it will be removed in "
                                  "future versions"}
            )
        };

        auto customParser = [](std::string_view value) -> std::optional<size_t>
        {
            auto lowerValue = utilities::toLowerCopy(value);
            std::erase(lowerValue, 'd');
            const auto dimensionality = utilities::stringToInt(lowerValue);

            // at the moment only 3-dimensional simulations are supported
            if (dimensionality == 3)
                return static_cast<size_t>(dimensionality);

            return std::nullopt;
        };

        const auto setValue = [](size_t /*value*/)
        {
            // do nothing as this key had always only one allowed value
            // (3-dimensional) and if it would not have been three it would have
            // even be used wrong -- key will be removed with version 2.0.0
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<size_t>{
                .metadata     = metaData,
                .customParser = customParser,
                .onSet        = setValue
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void GeneralInputParser::addFloatingPointTypeKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "floating_point_type",
            .title = "Floating Point Type",
            .description =
                "Specifies the floating point type for the simulation "
                "(float "
                "or double)"
        };

        const auto setValue = [this](FPType value)
        { _generalSettings->setFloatingPointTypeByUser(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<FPType>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void GeneralInputParser::addRandomSeedKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "random_seed",
            .title       = "Random Seed",
            .description = "Specifies the random seed for the simulation"
        };

        const auto setValue = [this](std::uint_fast32_t value)
        { _generalSettings->setRandomSeedByUser(value); };

        const auto validator = RangeValidator<std::uint_fast32_t>{
            0,
            std::numeric_limits<std::uint32_t>::max()
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::uint_fast32_t>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(validator)},
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

}   // namespace input
