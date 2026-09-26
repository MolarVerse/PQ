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

#include "resetKineticsInputParser.hpp"

#include <cstddef>   // for size_t, std

#include "inputKeyAdapter.hpp"
#include "resetKineticsSettings.hpp"   // for ResetKineticsSettings

using namespace input;
using namespace exc;
using namespace settings;

/**
 * @brief Construct a new Input File Parser Reset Kinetics:: Input File Parser
 * Reset Kinetics object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) nscale "<size_t>" 2) fscale
 * "<size_t>" 3) nreset "<size_t>" 4) freset "<size_t>"
 */
ResetKineticsInputParser::ResetKineticsInputParser()
{
    addNScaleKeyword();
    addFScaleKeyword();
    addNResetKeyword();
    addFResetKeyword();
    addNResetAngularKeyword();
    addFResetAngularKeyword();
    addFResetForcesKeyword();
}

/**
 * @brief add nscale keyword to the registry
 *
 * @details default value is 0
 */
void ResetKineticsInputParser::addNScaleKeyword()
{
    const auto metaData = KeyMetadata{
        .name  = "nscale",
        .title = "Number of steps for temperature reset",
        .description =
            "Specifies for how many steps at the beginning of the simulation "
            "the temperature is reset"
    };

    const auto setValue = [](size_t value)
    { ResetKineticsSettings::setNScale(value); };

    auto &keyword = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = 0,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(keyword), false);
}

/**
 * @brief add fscale keyword to the registry
 *
 * @details default value is 0 but then set to UINT_MAX in setup
 */
void ResetKineticsInputParser::addFScaleKeyword()
{
    const auto metaData = KeyMetadata{
        .name  = "fscale",
        .title = "Frequency of temperature reset",
        .description =
            "Specifies how frequently the temperature is reset during the "
            "simulation"
    };

    const auto setValue = [](size_t value)
    { ResetKineticsSettings::setFScale(value); };

    auto &keyword = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = 0,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(keyword), false);
}

/**
 * @brief add nreset keyword to the registry
 *
 * @details default value is 0
 */
void ResetKineticsInputParser::addNResetKeyword()
{
    const auto metaData = KeyMetadata{
        .name  = "nreset",
        .title = "Number of steps for momentum reset",
        .description =
            "Specifies for how many steps at the beginning of the simulation "
            "the momentum is reset"
    };

    const auto setValue = [](size_t value)
    { ResetKineticsSettings::setNReset(value); };

    auto &keyword = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = 0,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(keyword), false);
}

/**
 * @brief add freset keyword to the registry
 *
 * @details default value is 0 but then set to UINT_MAX in setup
 */
void ResetKineticsInputParser::addFResetKeyword()
{
    const auto metaData = KeyMetadata{
        .name  = "freset",
        .title = "Frequency of momentum reset",
        .description =
            "Specifies how frequently the momentum is reset during the "
            "simulation"
    };

    const auto setValue = [](size_t value)
    { ResetKineticsSettings::setFReset(value); };

    auto &keyword = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = 0,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(keyword), false);
}

/**
 * @brief add nreset_angular keyword to the registry
 *
 * @details default value is 0
 */
void ResetKineticsInputParser::addNResetAngularKeyword()
{
    const auto metaData = KeyMetadata{
        .name  = "nreset_angular",
        .title = "Number of steps for angular momentum reset",
        .description =
            "Specifies for how many steps at the beginning of the simulation "
            "the angular momentum is reset"
    };

    const auto setValue = [](size_t value)
    { ResetKineticsSettings::setNResetAngular(value); };

    auto &keyword = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = 0,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(keyword), false);
}

/**
 * @brief add freset_angular keyword to the registry
 *
 * @details default value is 0
 */
void ResetKineticsInputParser::addFResetAngularKeyword()
{
    const auto metaData = KeyMetadata{
        .name  = "freset_angular",
        .title = "Frequency of angular momentum reset",
        .description =
            "Specifies how frequently the angular momentum is reset during the "
            "simulation"
    };

    const auto setValue = [](size_t value)
    { ResetKineticsSettings::setFResetAngular(value); };

    auto &keyword = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = 0,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(keyword), false);
}

/**
 * @brief add freset_force keyword to the registry
 *
 * @details default value is 0
 */
void ResetKineticsInputParser::addFResetForcesKeyword()
{
    const auto metaData = KeyMetadata{
        .name  = "freset_forces",
        .title = "Frequency of force reset",
        .description =
            "Specifies how frequently the force is reset during the simulation"
    };

    const auto setValue = [](size_t value)
    { ResetKineticsSettings::setFResetForces(value); };

    auto &keyword = _getRegistry().registerKey(
        KeyRegistry<size_t>{
            .metadata     = metaData,
            .defaultValue = 0,
            .onSet        = setValue,
        }
    );

    addKeyword(metaData.name, adapt(keyword), false);
}
