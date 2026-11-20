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
#include <format>
#include <optional>

#include "enums/general.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "keyValidatorBase.hpp"
#include "parserUtils.hpp"
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
    GeneralInputParser::GeneralInputParser()
    {
        addKeyword(
            std::string("jobtype"),
            bindMember(&GeneralInputParser::parseJobType, this),
            true
        );

        addDimensionalityKey();
        addFloatingPointTypeKey();
        addRandomSeedKey();
    }

    /**
     * @brief parse jobtype of simulation left empty just to not parse it again
     * after engine is generated
     */
    void GeneralInputParser::parseJobType(
        const std::vector<std::string> & /*lineElements*/,
        size_t /*lineNumber*/
    )
    {
    }

    /**
     * @brief parse jobtype of simulation and set it in settings and reset
     * engine unique_ptr
     *
     * @details Possible options are:
     * 1) mm-md
     * 2) qm-md
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if jobtype is not recognised
     */
    void GeneralInputParser::parseJobTypeForEngine(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto jobtype =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        const auto jobTypeEnum =
            JobTypeMeta::from_stringCaseInsensitive(jobtype);

        if (!jobTypeEnum.has_value())
        {
            throw exc::InputFileException(format(
                "Invalid jobtype \"{}\" in input file - possible values are:\n"
                "- mm-opt\n"
                "- mm-hessian\n"
                "- mm-md\n"
                "- qm-md\n"
                "- qm-rpmd\n"
                "- qmmm-md\n",
                lineElements[2]
            ));
        }

        GeneralSettings::setJobtype(jobTypeEnum.value());
    }

    void GeneralInputParser::addDimensionalityKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "dim",
            .title = "Dimensionality",
            .description =
                "Specifies the dimensionality of the simulation (3 or 3d)",
            .errorMessage = "Dimensionality must be 3 or 3d"
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

        const auto setValue = [](size_t value)
        { GeneralSettings::setDimensionality(value); };

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
                "Specifies the floating point type for the simulation (float "
                "or double)"
        };

        const auto setValue = [](FPType value)
        { GeneralSettings::setFloatingPointType(value); };

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

        const auto setValue = [](std::uint_fast32_t value)
        {
            GeneralSettings::setRandomSeed(value);
            GeneralSettings::setIsRandomSeedSet(true);
        };

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
