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

#include <cstddef>      // for size_t
#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "engine.hpp"              // for Engine
#include "exceptions.hpp"          // for InputFileException, customException
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox
#include "simulationBoxSettings.hpp"   // for setDensitySet
#include "stringUtilities.hpp"         // for toLowerCopy

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;

/**
 * @brief Construct a new Input File Parser Simulation Box:: Input File Parser
 * Simulation Box object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) rcoulomb <double> 2) density
 * <double>
 *
 * @param engine
 */
SimulationBoxInputParser::SimulationBoxInputParser(Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("rcoulomb"),
        bind_front(&SimulationBoxInputParser::parseCoulombRadius, this),
        false
    );
    addKeyword(
        std::string("density"),
        bind_front(&SimulationBoxInputParser::parseDensity, this),
        false
    );
    addKeyword(
        std::string("init_velocities"),
        bind_front(&SimulationBoxInputParser::parseInitializeVelocities, this),
        false
    );
}

/**
 * @brief parses the coulomb cutoff radius
 *
 * @details default value is 12.5
 *
 * @param lineElements
 *
 * @throw InputFileException if the cutoff radius is negative
 */
void SimulationBoxInputParser::parseCoulombRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto cutOff = stod(lineElements[2]);

    if (cutOff < 0.0)
        throw InputFileException(format(
            "Coulomb radius cutoff must be positive - \"{}\" at line {} in "
            "input file",
            lineElements[2],
            lineNumber
        ));

    PotentialSettings::setCoulombRadiusCutOff(cutOff);
}

/**
 * @brief parse density of simulation and set it in simulation box
 *
 * @details set in simulationBoxSettings if density is set to put warning if
 * both density and box size are set
 *
 * @param lineElements
 *
 * @throw InputFileException if the density is negative
 */
void SimulationBoxInputParser::parseDensity(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto density = stod(lineElements[2]);

    if (density < 0.0)
        throw InputFileException(
            std::format("Density must be positive - density = {}", density)
        );

    SimulationBoxSettings::setDensitySet(true);
    _engine.getSimulationBox().setDensity(density);
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
    const size_t                    lineNumber
)
{
    using enum InitVelocities;
    checkCommand(lineElements, lineNumber);

    const auto initializeVelocities = toLowerCopy(lineElements[2]);

    if (initializeVelocities == "true")
        SimulationBoxSettings::setInitializeVelocities(TRUE);

    else if (initializeVelocities == "false")
        SimulationBoxSettings::setInitializeVelocities(FALSE);

    else if (initializeVelocities == "force")
        SimulationBoxSettings::setInitializeVelocities(FORCE);

    else
        throw InputFileException(std::format(
            "Invalid value for initialize velocities - \"{}\" at line {} in "
            "input file.\n"
            "Possible options are: true, false, force",
            lineElements[2],
            lineNumber
        ));
}