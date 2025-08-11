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

#include "hybridSetup.hpp"

#include <format>   // for format
#include <string>   // for string
#include <vector>   // for vector

#include "engine.hpp"           // for Engine
#include "exceptions.hpp"       // for InputFileException, UserInputException
#include "hybridSettings.hpp"   // for HybridSettings
#include "settings.hpp"         // for Settings

using setup::HybridSetup;
using namespace settings;
using namespace engine;
using namespace customException;

/**
 * @brief wrapper to build HybridSetup object and call setup
 *
 * @param engine
 */
void setup::setupHybrid(Engine &engine)
{
    if (!Settings::isHybridJobtype())
        return;

    engine.getStdoutOutput().writeSetup("Hybrid setup");
    engine.getLogOutput().writeSetup("Hybrid setup");

    HybridSetup hybridSetup(engine);
    hybridSetup.setup();
}

/**
 * @brief Construct a new HybridSetup object
 *
 * @param engine
 */
HybridSetup::HybridSetup(Engine &engine) : _engine(engine) {}

/**
 * @brief setup Hybrid-MD
 *
 */
void HybridSetup::setup()
{
    setupInnerRegionCenter();
    setupForcedInnerList();
    setupForcedOuterList();
    checkZoneRadii();
    throw UserInputException("Not implemented yet");
}

/**
 * @brief setup inner region center
 *
 * @details This function determines the indices of the atoms that mark the
 * center of the inner region of hybrid type calculations. All atomIndices
 * that are part of the inner region center are added to the inner region center
 * list in the simulation box.
 *
 */
void HybridSetup::setupInnerRegionCenter()
{
    _engine.getSimulationBox().addInnerRegionCenterAtoms(
        HybridSettings::getInnerRegionCenter()
    );
}

/**
 * @brief setup forced inner list
 *
 */
void HybridSetup::setupForcedInnerList()
{
    _engine.getSimulationBox().setupForcedInnerAtoms(
        HybridSettings::getForcedInnerList()
    );
}

/**
 * @brief setup forced outer list
 *
 */
void HybridSetup::setupForcedOuterList()
{
    _engine.getSimulationBox().setupForcedOuterAtoms(
        HybridSettings::getForcedOuterList()
    );
}

/**
 * @brief Validate zone radii configuration for hybrid calculations
 *
 * @throws customException::InputFileException if the core radius is larger than
 * the layer radius
 * @throws customException::InputFileException if the smoothing region is too
 * thick for the chosen combinatin of core and layer radius
 */
void HybridSetup::checkZoneRadii()
{
    const auto coreRadius  = HybridSettings::getCoreRadius();
    const auto layerRadius = HybridSettings::getLayerRadius();
    const auto smoothingRegionThickness =
        HybridSettings::getSmoothingRegionThickness();

    if (coreRadius > layerRadius)
        throw(InputFileException(
            std::format(
                "Core radius ({} Å) cannot be larger than layer radius ({} Å)",
                coreRadius,
                layerRadius
            )
        ));

    if (coreRadius > (layerRadius - smoothingRegionThickness))
        throw(InputFileException(
            std::format(
                "Smoothing region is too thick ({} Å) for the chosen "
                "combination of core ({} Å) and layer radius ({} Å)",
                smoothingRegionThickness,
                coreRadius,
                layerRadius
            )
        ));
}