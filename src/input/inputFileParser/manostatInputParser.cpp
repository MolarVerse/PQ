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
#include <limits>        // for numeric_limits
#include <string_view>   // for string_view

#include "constants/conversionFactors.hpp"
#include "exceptions.hpp"   // for exc::InputFileException, customException
#include "manostatSettings.hpp"   // for settings::ManostatSettings
#include "parserUtils.hpp"
#include "references.hpp"         // for references::ReferencesOutput
#include "referencesOutput.hpp"   // for references::ReferencesOutput
#include "stringUtilities.hpp"    // for toLowerCopy

namespace input
{

    /**
     * @brief Construct a new Input File Parser Manostat:: Input File Parser
     * Manostat object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) manostat "<string>" 2)
     * pressure
     * "<double>" (only required if manostat is not none) 3) p_relaxation
     * "<double>" 4) compressibility "<double>"
     */
    ManostatInputParser::ManostatInputParser()
    {
        addKeyword(
            std::string("manostat"),
            bindMember(&ManostatInputParser::parseManostat, this),
            false
        );

        addKeyword(
            std::string("pressure"),
            bindMember(&ManostatInputParser::parsePressure, this),
            false
        );

        addKeyword(
            std::string("p_relaxation"),
            bindMember(&ManostatInputParser::parseManostatRelaxationTime, this),
            false
        );

        addKeyword(
            std::string("compressibility"),
            bindMember(&ManostatInputParser::parseCompressibility, this),
            false
        );

        addKeyword(
            std::string("isotropy"),
            bindMember(&ManostatInputParser::parseIsotropy, this),
            false
        );

        addKeyword(
            std::string("fixed_axis"),
            bindMember(&ManostatInputParser::parseFixedAxis, this),
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
     * @param lineNumber
     *
     * @throws exc::InputFileException if manostat is not berendsen or
     * none
     */
    void ManostatInputParser::parseManostat(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto manostat =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        using enum settings::ManostatType;

        if (manostat == "none")
            settings::ManostatSettings::setManostatType(NONE);

        else if (manostat == "berendsen")
        {
            settings::ManostatSettings::setManostatType(BERENDSEN);
            references::ReferencesOutput::addReferenceFile(
                references::BERENDSEN_FILE
            );
        }

        else if (manostat == "stochastic_rescaling")
        {
            settings::ManostatSettings::setManostatType(STOCHASTIC_RESCALING);
            references::ReferencesOutput::addReferenceFile(
                references::STOCHASTIC_RESCALING_FILE
            );
        }

        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid manostat \"{}\" at line {} in input file.\n"
                    "Possible options are: berendsen, stochastic_rescaling and "
                    "none",
                    lineElements[2],
                    lineNumber
                )
            );
        }
    }

    /**
     * @brief Parse the pressure used in the simulation
     *
     * @details no default value - if needed it has to be set in the input file
     *
     * @param lineElements
     * @param lineNumber
     */
    void ManostatInputParser::parsePressure(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto pressure = utilities::stringToFiniteDouble(lineElements[2]);

        settings::ManostatSettings::setTargetPressure(pressure);
    }

    /**
     * @brief parses the relaxation time of the manostat
     *
     * @details default value is 1.0
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if relaxation time is negative
     */
    void ManostatInputParser::parseManostatRelaxationTime(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        const auto relaxationTime =
            utilities::stringToFiniteDouble(lineElements[2]);

        if (relaxationTime <= 0.0)
        {
            throw exc::InputFileException(
                "Relaxation time of manostat must be finite and greater than "
                "zero"
            );
        }

        if (relaxationTime > std::numeric_limits<double>::max() / PS_TO_FS)
        {
            throw exc::InputFileException(
                "Relaxation time of manostat is too large to represent in "
                "femtoseconds"
            );
        }

        settings::ManostatSettings::setTauManostat(relaxationTime);
    }

    /**
     * @brief Parse the compressibility used in the simulation (isothermal
     * compressibility)
     *
     * @details default value is 4.5e-5
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if compressibility is negative
     */
    void ManostatInputParser::parseCompressibility(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);
        const auto compressibility =
            utilities::stringToFiniteDouble(lineElements[2]);

        if (compressibility < 0.0)
            throw exc::InputFileException(
                "Compressibility must be finite and non-negative"
            );

        settings::ManostatSettings::setCompressibility(compressibility);
    }

    /**
     * @brief Parse the isotropy of the manostat
     *
     * @details Possible options are:
     * 1) "isotropic"                        - isotropic manostat is used
     * (default) 2) "xy", "yx", "xz", "zx", "yz", "zy" - semi isotropic manostat
     * is used 3) "anisotropic"                      - anisotropic manostat is
     * used
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if isotropy is not isotropic,
     * semi_isotropic or anisotropic
     */
    void ManostatInputParser::parseIsotropy(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto isotropy =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        using enum settings::Isotropy;

        if (isotropy == "isotropic")
            settings::ManostatSettings::setIsotropy(ISOTROPIC);

        else if (isotropy == "xy" || isotropy == "yx")
        {
            settings::ManostatSettings::setIsotropy(SEMI_ISOTROPIC);
            settings::ManostatSettings::set2DIsotropicAxes({0, 1});
            settings::ManostatSettings::set2DAnisotropicAxis(2);
        }

        else if (isotropy == "xz" || isotropy == "zx")
        {
            settings::ManostatSettings::setIsotropy(SEMI_ISOTROPIC);
            settings::ManostatSettings::set2DIsotropicAxes({0, 2});
            settings::ManostatSettings::set2DAnisotropicAxis(1);
        }

        else if (isotropy == "yz" || isotropy == "zy")
        {
            settings::ManostatSettings::setIsotropy(SEMI_ISOTROPIC);
            settings::ManostatSettings::set2DIsotropicAxes({1, 2});
            settings::ManostatSettings::set2DAnisotropicAxis(0);
        }

        else if (isotropy == "anisotropic")
        {
            settings::ManostatSettings::setIsotropy(ANISOTROPIC);
        }
        else if (isotropy == "full_anisotropic")
        {
            settings::ManostatSettings::setIsotropy(FULL_ANISOTROPIC);
        }
        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid isotropy \"{}\" at line {} in input file.\n"
                    "Possible options are: isotropic, xy, xz, yz, "
                    "anisotropic and full_anisotropic",
                    lineElements[2],
                    lineNumber
                )
            );
        }
    }

    void ManostatInputParser::parseFixedAxis(
        const std::vector<std::string> &lineElements,
        const size_t                    lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto fixed_axis =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        using enum settings::FixedAxis;

        if (fixed_axis == "none")
            settings::ManostatSettings::setFixedAxis(NONE);

        else if (fixed_axis == "x")
            settings::ManostatSettings::setFixedAxis(X);

        else if (fixed_axis == "y")
            settings::ManostatSettings::setFixedAxis(Y);

        else if (fixed_axis == "z")
            settings::ManostatSettings::setFixedAxis(Z);

        else if (fixed_axis == "xy" || fixed_axis == "yx")
            settings::ManostatSettings::setFixedAxis(XY);

        else if (fixed_axis == "xz" || fixed_axis == "zx")
            settings::ManostatSettings::setFixedAxis(XZ);

        else if (fixed_axis == "yz" || fixed_axis == "zy")
            settings::ManostatSettings::setFixedAxis(YZ);

        else if (fixed_axis == "all" || fixed_axis == "xyz")
            settings::ManostatSettings::setFixedAxis(ALL);

        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid fixed_axis \"{}\" at line {} in input file.\n"
                    "Possible options are: none, x, y, z, xy, xz, yz, all",
                    lineElements[2],
                    lineNumber
                )
            );
        }
    }

}   // namespace input
