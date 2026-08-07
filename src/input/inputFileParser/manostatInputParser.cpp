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

#include "manostatInputParser.hpp"

#include <cstddef>       // for size_t
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

#include "exceptions.hpp"         // for InputFileException, customException
#include "manostatSettings.hpp"   // for ManostatSettings
#include "references.hpp"         // for ReferencesOutput
#include "referencesOutput.hpp"   // for ReferencesOutput
#include "stringUtilities.hpp"    // for toLowerCopy

using namespace input;
using namespace engine;
using namespace settings;
using namespace customException;
using namespace references;
using namespace utilities;

/**
 * @brief Construct a new Input File Parser Manostat:: Input File Parser
 * Manostat object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) manostat <string> 2) pressure
 * <double> (only required if manostat is not none) 3) p_relaxation <double> 4)
 * compressibility <double>
 *
 * @param engine
 */
ManostatInputParser::ManostatInputParser(Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("manostat"),
        bind_front(&ManostatInputParser::parseManostat, this),
        false
    );

    addKeyword(
        std::string("pressure"),
        bind_front(&ManostatInputParser::parsePressure, this),
        false
    );

    addKeyword(
        std::string("p_relaxation"),
        bind_front(&ManostatInputParser::parseManostatRelaxationTime, this),
        false
    );

    addKeyword(
        std::string("compressibility"),
        bind_front(&ManostatInputParser::parseCompressibility, this),
        false
    );

    addKeyword(
        std::string("isotropy"),
        bind_front(&ManostatInputParser::parseIsotropy, this),
        false
    );
}

/**
 * @brief Parse the manostat used in the simulation
 *
 * @details Possible options are:
 * 1) "none"                 - no manostat is used (default)
 * 2) "berendsen"            - berendsen manostat is used
 * 3) "stochastic_rescaling" - stochastic rescaling manostat is used
 *
 * @param lineElements
 *
 * @throws InputFileException if manostat is not berendsen or
 * none
 */
void ManostatInputParser::parseManostat(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto manostat = toLowerAndReplaceDashesCopy(lineElements[2]);

    using enum ManostatType;

    if (manostat == "none")
        ManostatSettings::setManostatType(NONE);

    else if (manostat == "berendsen")
    {
        ManostatSettings::setManostatType(BERENDSEN);
        ReferencesOutput::addReferenceFile(_BERENDSEN_FILE_);
    }

    else if (manostat == "stochastic_rescaling")
    {
        ManostatSettings::setManostatType(STOCHASTIC_RESCALING);
        ReferencesOutput::addReferenceFile(_STOCHASTIC_RESCALING_FILE_);
    }

    else
        throw InputFileException(std::format(
            "Invalid manostat \"{}\" at line {} in input file.\n"
            "Possible options are: berendsen, stochastic_rescaling and none",
            lineElements[2],
            lineNumber
        ));
}

/**
 * @brief Parse the pressure used in the simulation
 *
 * @details no default value - if needed it has to be set in the input file
 *
 * @param lineElements
 */
void ManostatInputParser::parsePressure(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    ManostatSettings::setTargetPressure(stod(lineElements[2]));
    ManostatSettings::setPressureSet(true);
}

/**
 * @brief parses the relaxation time of the manostat
 *
 * @details default value is 1.0
 *
 * @param lineElements
 *
 * @throw InputFileException if relaxation time is negative
 */
void ManostatInputParser::parseManostatRelaxationTime(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    const auto relaxationTime = stod(lineElements[2]);

    if (relaxationTime < 0)
        throw InputFileException(
            "Relaxation time of manostat cannot be negative"
        );

    ManostatSettings::setTauManostat(relaxationTime);
}

/**
 * @brief Parse the compressibility used in the simulation (isothermal
 * compressibility)
 *
 * @details default value is 4.5e-5
 *
 * @param lineElements
 *
 * @throw InputFileException if compressibility is negative
 */
void ManostatInputParser::parseCompressibility(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    const auto compressibility = stod(lineElements[2]);

    if (compressibility < 0.0)
        throw InputFileException("Compressibility cannot be negative");

    ManostatSettings::setCompressibility(compressibility);
}

/**
 * @brief Parse the isotropy of the manostat
 *
 * @details Possible options are:
 * 1) "isotropic"                        - isotropic manostat is used (default)
 * 2) "xy", "yx", "xz", "zx", "yz", "zy" - semi isotropic manostat is used
 * 3) "anisotropic"                      - anisotropic manostat is used
 *
 * @param lineElements
 *
 * @throws InputFileException if isotropy is not isotropic,
 * semi_isotropic or anisotropic
 */
void ManostatInputParser::parseIsotropy(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto isotropy = toLowerAndReplaceDashesCopy(lineElements[2]);

    using enum Isotropy;

    if (isotropy == "isotropic")
        ManostatSettings::setIsotropy(ISOTROPIC);

    else if (isotropy == "xy" || isotropy == "yx")
    {
        ManostatSettings::setIsotropy(SEMI_ISOTROPIC);
        ManostatSettings::set2DIsotropicAxes({0, 1});
        ManostatSettings::set2DAnisotropicAxis(2);
    }

    else if (isotropy == "xz" || isotropy == "zx")
    {
        ManostatSettings::setIsotropy(SEMI_ISOTROPIC);
        ManostatSettings::set2DIsotropicAxes({0, 2});
        ManostatSettings::set2DAnisotropicAxis(1);
    }

    else if (isotropy == "yz" || isotropy == "zy")
    {
        ManostatSettings::setIsotropy(SEMI_ISOTROPIC);
        ManostatSettings::set2DIsotropicAxes({1, 2});
        ManostatSettings::set2DAnisotropicAxis(0);
    }

    else if (isotropy == "anisotropic")
        ManostatSettings::setIsotropy(ANISOTROPIC);

    else if (isotropy == "full_anisotropic")
        ManostatSettings::setIsotropy(FULL_ANISOTROPIC);

    else
        throw InputFileException(std::format(
            "Invalid isotropy \"{}\" at line {} in input file.\n"
            "Possible options are: isotropic, xy, xz, yz, "
            "anisotropic and full_anisotropic",
            lineElements[2],
            lineNumber
        ));
}