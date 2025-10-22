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

#include "MMInputParser.hpp"

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "engine.hpp"                 // for Engine
#include "exceptions.hpp"             // for InputFileException, customException
#include "forceFieldClass.hpp"        // for ForceField
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "forceFieldSettings.hpp"     // for ForceFieldSettings
#include "potential.hpp"              // for Potential
#include "potentialSettings.hpp"      // for PotentialSettings
#include "stringUtilities.hpp"        // for toLowerCopy
#include "waterModelSettings.hpp"     // for WaterModelSettings

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;
using namespace potential;

/**
 * @brief Construct a new Input File Parser Force Field:: Input File Parser
 * Force Field object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) force-field <on/off/bonded>
 *
 * @param engine
 */
MMInputParser::MMInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("force-field"),
        bind_front(&MMInputParser::parseForceFieldType, this),
        false
    );
    addKeyword(
        std::string("noncoulomb"),
        bind_front(&MMInputParser::parseNonCoulombType, this),
        false
    );
    addKeyword(
        std::string("water_intra"),
        bind_front(&MMInputParser::parseWaterIntraModel, this),
        false
    );
    addKeyword(
        std::string("water_inter"),
        bind_front(&MMInputParser::parseWaterInterModel, this),
        false
    );
}

/**
 * @brief Parse the force field type
 *
 * @details Possible options are:
 * 1) "on"  - force-field is activated
 * 2) "off" - force-field is deactivated (default)
 * 3) "bonded" - only bonded interactions are activated
 *
 * @param lineElements
 *
 * @throws InputFileException if force-field is not valid - currently only on,
 * off and bonded are supported
 */
void MMInputParser::parseForceFieldType(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto forceFieldType = toLowerCopy(lineElements[2]);

    if (forceFieldType == "on")
    {
        ForceFieldSettings::activate();
        _engine.getForceFieldPtr()->activateNonCoulombic();
        _engine.getPotential().makeNonCoulombPotential(ForceFieldNonCoulomb());
    }
    else if (forceFieldType == "off")
    {
        ForceFieldSettings::deactivate();
        _engine.getForceFieldPtr()->deactivateNonCoulombic();
    }
    else if (forceFieldType == "bonded")
    {
        ForceFieldSettings::activate();
        _engine.getForceFieldPtr()->deactivateNonCoulombic();
    }
    else
        throw InputFileException(format(
            "Invalid force-field keyword \"{}\" at line {} "
            "in input file\n"
            "Possible options are \"on\", \"off\" or \"bonded\"",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parse the nonCoulombic type of the guff.dat file
 *
 * @details Possible options are:
 * 1) "guff"  - guff.dat file is used (default)
 * 2) "lj"
 * 3) "buck"
 * 4) "morse"
 *
 * @param lineElements
 *
 * @throws InputFileException if invalid nonCoulomb type
 */
void MMInputParser::parseNonCoulombType(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto type = toLowerCopy(lineElements[2]);

    using enum NonCoulombType;

    if (type == "guff")
        PotentialSettings::setNonCoulombType(GUFF);

    else if (type == "lj")
        PotentialSettings::setNonCoulombType(LJ);

    else if (type == "buck")
        PotentialSettings::setNonCoulombType(BUCKINGHAM);

    else if (type == "morse")
        PotentialSettings::setNonCoulombType(MORSE);

    else
        throw InputFileException(format(
            "Invalid nonCoulomb type \"{}\" at line {} in input file.\n"
            "Possible options are: lj, buck, morse and guff",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parse the intramolecular water model type
 *
 * @details Possible options are:
 * 1) "SPC/Fw" - SPC flexible water model for intramolecular interactions
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if invalid water_intra model type
 */
void MMInputParser::parseWaterIntraModel(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum WaterIntraModel;

    checkCommand(lineElements, lineNumber);

    const auto waterIntraModel = toLowerAndReplaceDashesCopy(lineElements[2]);

    if (waterIntraModel == "spc/fw")
        WaterModelSettings::setWaterIntraModel(SPC_FW);
    else
        throw InputFileException(format(
            "Invalid water_intra keyword \"{}\" at line {} "
            "in input file\n"
            "Possible options are \"SPC/Fw\"",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parse the intermolecular water model type
 *
 * @details Possible options are:
 * 1) "SPC/Fw" - SPC flexible water model for intermolecular interactions
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if invalid water_inter model type
 */
void MMInputParser::parseWaterInterModel(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    using enum WaterInterModel;

    checkCommand(lineElements, lineNumber);

    const auto waterInterModel = toLowerAndReplaceDashesCopy(lineElements[2]);

    if (waterInterModel == "spc/fw")
        WaterModelSettings::setWaterInterModel(SPC_FW);
    else
        throw InputFileException(format(
            "Invalid water_inter keyword \"{}\" at line {} "
            "in input file\n"
            "Possible options are \"SPC/Fw\"",
            lineElements[2],
            lineNumber
        ));
}
