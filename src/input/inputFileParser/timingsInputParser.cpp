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

#include "timingsInputParser.hpp"

#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "rangeValidator.hpp"
#include "timingsSettings.hpp"   // for TimingsSettings

using namespace input;
using namespace exc;
using namespace settings;

/**
 * @brief Construct a new Input File Parser Timings object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) timestep "<double>" (required)
 * 2) nstep "<size_t>" (required)
 */
TimingsInputParser::TimingsInputParser()
{
    addTimeStep();
    addNumberOfSteps();
}

/**
 * @brief Add the timestep key to the input parser
 *
 * @details This function registers the "timestep" keyword with the input
 * parser, including its metadata, validation, and callback to set the value in
 * TimingsSettings.
 */
void TimingsInputParser::addTimeStep()
{
    const auto metaData = KeyMetadata{
        .name        = "timestep",
        .title       = "Timestep of the simulation",
        .description = "The time step used in the simulation",
        .unit        = "fs"
    };

    const RangeValidator<double, Greater::GT> validator{0.0, std::nullopt};

    const auto setTimeStep = [&](double value)
    { TimingsSettings::setTimeStep(value); };

    auto &timeStep = _getRegistry().registerKey(
        KeyRegistry<double>{
            .metadata  = metaData,
            .onSet     = setTimeStep,
            .validator = makeShared(validator)
        }
    );

    addKeyword(std::string("timestep"), adapt(timeStep), false);
}

/**
 * @brief Add the number of steps key to the input parser
 *
 * @details This function registers the "nstep" keyword with the input parser,
 * including its metadata, validation, and callback to set the value in
 * TimingsSettings.
 */
void TimingsInputParser::addNumberOfSteps()
{
    const auto metaData = KeyMetadata{
        .name        = "nstep",
        .title       = "Number of steps of the simulation",
        .description = "The total number of steps in the simulation"
    };

    const RangeValidator<int> validator{1, std::nullopt};

    const auto setNumberOfSteps = [&](int value)
    { TimingsSettings::setNumberOfSteps(static_cast<size_t>(value)); };

    auto &numberOfSteps = _getRegistry().registerKey(
        KeyRegistry<int>{
            .metadata  = metaData,
            .onSet     = setNumberOfSteps,
            .validator = std::make_shared<RangeValidator<int>>(validator)
        }
    );

    addKeyword(std::string("nstep"), adapt(numberOfSteps), false);
}
