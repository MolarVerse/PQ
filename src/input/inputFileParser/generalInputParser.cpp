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

#include <cstdint>     // for uint_fast32_t and UINT32_MAX
#include <format>      // for format
#include <stdexcept>   // for out_of_range and invalid_argument

#include "engine.hpp"          // for Engine
#include "exceptions.hpp"      // for exc::InputFileException, customException
#include "hessianEngine.hpp"   // for HessianEngine
#include "mmmdEngine.hpp"      // for MMMDEngine
#include "optEngine.hpp"       // for MMOptEngine
#include "parserUtils.hpp"
#include "qmmdEngine.hpp"              // for QMMDEngine
#include "qmmmMDEngine.hpp"            // for QMMMMDEngine
#include "ringPolymerqmmdEngine.hpp"   // for RingPolymerQMMDEngine
#include "settings.hpp"                // for settings::Settings
#include "stringUtilities.hpp"         // for toLowerCopy

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

        addKeyword(
            std::string("dim"),
            bindMember(&GeneralInputParser::parseDimensionality, this),
            false
        );

        addKeyword(
            std::string("floating_point_type"),
            bindMember(&GeneralInputParser::parseFloatingPointType, this),
            false
        );

        addKeyword(
            std::string("random_seed"),
            bindMember(&GeneralInputParser::parseRandomSeed, this),
            false
        );
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
     * @param engine
     *
     * @throw exc::InputFileException if jobtype is not recognised
     */
    void GeneralInputParser::parseJobTypeForEngine(
        const std::vector<std::string>  &lineElements,
        size_t                           lineNumber,
        std::unique_ptr<engine::Engine> &engine
    )
    {
        using enum settings::JobType;
        checkCommand(lineElements, lineNumber);

        const auto jobtype =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        if (jobtype == "mm_opt")
        {
            settings::Settings::setJobtype(MM_OPT);
            engine = std::make_unique<engine::OptEngine>();
        }
        else if (jobtype == "mm_hessian")
        {
            settings::Settings::setJobtype(MM_HESSIAN);
            engine = std::make_unique<engine::HessianEngine>();
        }
        else if (jobtype == "mm_md")
        {
            settings::Settings::setJobtype(MM_MD);
            engine = std::make_unique<engine::MMMDEngine>();
        }
        else if (jobtype == "qm_md")
        {
            settings::Settings::setJobtype(QM_MD);
            engine = std::make_unique<engine::QMMDEngine>();
        }
        else if (jobtype == "qm_rpmd")
        {
            settings::Settings::setJobtype(RING_POLYMER_QM_MD);
            engine = std::make_unique<engine::RingPolymerQMMDEngine>();
        }
        else if (jobtype == "qmmm_md")
        {
            settings::Settings::setJobtype(QMMM_MD);
            engine = std::make_unique<engine::QMMMMDEngine>();
        }
        else
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
    }

    /**
     * @brief parse dimensionality of simulation
     *
     * @details Possible options are:
     * 1) 3
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if dimensionality is not
     * recognised
     */
    void GeneralInputParser::parseDimensionality(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        auto dimensionalityString = utilities::toLowerCopy(lineElements[2]);

        std::erase(dimensionalityString, 'd');

        const auto dimensionality =
            utilities::stringToInt(dimensionalityString);

        if (dimensionality == 3)
        {
            settings::Settings::setDimensionality(
                static_cast<size_t>(dimensionality)
            );
        }
        else
        {
            throw exc::InputFileException(format(
                "Invalid dimensionality \"{}\" in input file\n"
                "Possible values are: 3, 3d",
                lineElements[2]
            ));
        }
    }

    /**
     * @brief parse floating point type of simulation
     *
     * @details Possible options are:
     * 1) float
     * 2) double
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if floating point type is not
     * recognised
     */
    void GeneralInputParser::parseFloatingPointType(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        using enum settings::FPType;
        checkCommand(lineElements, lineNumber);

        const auto floatingPointType = utilities::toLowerCopy(lineElements[2]);

        if (floatingPointType == "float")
            settings::Settings::setFloatingPointType(FLOAT);

        else if (floatingPointType == "double")
            settings::Settings::setFloatingPointType(DOUBLE);

        else
        {
            throw exc::InputFileException(format(
                "Invalid floating point type \"{}\" in input file\n"
                "Possible values are: float, double",
                lineElements[2]
            ));
        }
    }

    /**
     * @brief parse random seed value for PRNG
     *
     * @details value not set as default
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if random seed value is invalid,
     * negative, or exceeds uint_fast32_t range
     */
    void GeneralInputParser::parseRandomSeed(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        constexpr auto maxRandomSeed = static_cast<std::int64_t>(UINT32_MAX);

        auto throwRangeError = [&maxRandomSeed](const auto &value)
        {
            throw exc::InputFileException(format(
                "Random seed value \"{}\" is out of range.\n"
                "Must be an integer between \"0\" and \"{}\" (inclusive)",
                value,
                maxRandomSeed
            ));
        };

        auto throwValidityError = [&maxRandomSeed](const auto &value)
        {
            throw exc::InputFileException(format(
                "Random seed value \"{}\" is invalid.\n"
                "Must be an integer between \"0\" and \"{}\" (inclusive)",
                value,
                maxRandomSeed
            ));
        };

        std::uint_fast32_t randomSeed = 0;

        try
        {
            randomSeed = utilities::stringToUintFast32t(lineElements[2]);
        }
        catch (const std::invalid_argument &)
        {
            throwValidityError(lineElements[2]);
        }
        catch (const std::out_of_range &)
        {
            throwRangeError(lineElements[2]);
        }

        settings::Settings::setIsRandomSeedSet(true);
        settings::Settings::setRandomSeed(randomSeed);
    }

}   // namespace input
