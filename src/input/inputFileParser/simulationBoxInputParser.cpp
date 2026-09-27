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

#include "simulationBoxInputParser.hpp"

#include <cstddef>   // for size_t
#include <format>    // for format
#include <utility>

#include "exceptions.hpp"   // for exc::InputFileException, customException
#include "parserUtils.hpp"
#include "potentialSettings.hpp"   // for settings::PotentialSettings
#include "simulationBox.hpp"
#include "simulationBoxSettings.hpp"   // for setDensitySet
#include "stringUtilities.hpp"         // for toLowerCopy

namespace input
{

    /**
     * @brief Construct a new Input File Parser Simulation Box:: Input File
     * Parser Simulation Box object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) rcoulomb "<double>" 2)
     * density
     * "<double>"
     *
     * @param simulationBox
     */
    SimulationBoxInputParser::SimulationBoxInputParser(
        std::shared_ptr<molsys::SimulationBox> simulationBox
    )
        : _simulationBox(std::move(simulationBox))
    {
        addKeyword(
            std::string("rcoulomb"),
            bindMember(&SimulationBoxInputParser::parseCoulombRadius, this),
            false
        );
        addKeyword(
            std::string("rnoncoulomb"),
            bindMember(&SimulationBoxInputParser::parseNonCoulombRadius, this),
            false
        );
        addKeyword(
            std::string("density"),
            bindMember(&SimulationBoxInputParser::parseDensity, this),
            false
        );
        addKeyword(
            std::string("init_velocities"),
            bindMember(
                &SimulationBoxInputParser::parseInitializeVelocities,
                this
            ),
            false
        );
    }

    /**
     * @brief parses the coulomb cutoff radius
     *
     * @details default value is 12.5
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if the cutoff radius is negative
     */
    void SimulationBoxInputParser::parseCoulombRadius(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto cutOff = utilities::stringToFiniteDouble(lineElements[2]);

        if (cutOff < 0.0)
        {
            throw exc::InputFileException(format(
                "Coulomb radius cutoff must be positive - \"{}\" at line {} in "
                "input file",
                lineElements[2],
                lineNumber
            ));
        }

        settings::PotentialSettings::setCoulombRadiusCutOff(cutOff);
    }

    /**
     * @brief parses the non-coulomb cutoff radius
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if the cutoff radius is negative
     */
    void SimulationBoxInputParser::parseNonCoulombRadius(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto cutOff = stod(lineElements[2]);

        if (cutOff < 0.0)
        {
            throw exc::InputFileException(format(
                "Non-Coulomb radius cutoff must be positive - \"{}\" at line "
                "{} in "
                "input file",
                lineElements[2],
                lineNumber
            ));
        }

        settings::PotentialSettings::setNonCoulombRadiusCutOff(cutOff);
    }

    /**
     * @brief parse density of simulation and set it in simulation box
     *
     * @details set in simulationBoxSettings if density is set to put warning if
     * both density and box size are set
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throw exc::InputFileException if the density is negative
     */
    void SimulationBoxInputParser::parseDensity(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto density = utilities::stringToFiniteDouble(lineElements[2]);

        if (density <= 0.0)
            throw exc::InputFileException(
                std::format("Density must be positive - density = {}", density)
            );

        settings::SimulationBoxSettings::setDensitySet(true);
        _simulationBox->setDensity(density);
    }

    /**
     * @brief parse if velocities should be initialized with maxwell boltzmann
     * distribution
     *
     * @details possible options are:
     * 1) true
     * 2) false (default)
     * 3) force
     *
     * @param lineElements
     * @param lineNumber
     */
    void SimulationBoxInputParser::parseInitializeVelocities(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        using enum settings::InitVelocities;
        checkCommand(lineElements, lineNumber);

        const auto initializeVelocities =
            utilities::toLowerCopy(lineElements[2]);

        if (initializeVelocities == "true")
            settings::SimulationBoxSettings::setInitializeVelocities(TRUE);

        else if (initializeVelocities == "false")
            settings::SimulationBoxSettings::setInitializeVelocities(FALSE);

        else if (initializeVelocities == "force")
            settings::SimulationBoxSettings::setInitializeVelocities(FORCE);

        else
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid value for initialize velocities - \"{}\" at line "
                    "{} "
                    "in "
                    "input file.\n"
                    "Possible options are: true, false, force",
                    lineElements[2],
                    lineNumber
                )
            );
        }
    }

}   // namespace input
