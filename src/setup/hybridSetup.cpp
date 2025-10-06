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
}

/**
 * @brief setup inner region center
 *
 * @details This function determines the indices of the atoms that mark the
 * center of the inner region of hybrid type calculations. If inner region
 * center atoms are specified in the settings, those atom indices are used.
 * If no inner region center is specified, atom 0 (e.g. the first atom) is used
 * as the default center. All determined atom indices are added to the inner
 * region center list in the simulation box.
 *
 */
void HybridSetup::setupInnerRegionCenter()
{
    const auto innerRegionCenter = HybridSettings::getInnerRegionCenter();

    _engine.getSimulationBox().addInnerRegionCenterAtoms(
        innerRegionCenter ? innerRegionCenter.value() : std::vector<int>{0}
    );
}

/**
 * @brief setup forced inner list
 *
 */
void HybridSetup::setupForcedInnerList()
{
    _engine.getSimulationBox().setupForcedInnerMolecules(
        HybridSettings::getForcedInnerList()
    );
}

/**
 * @brief setup forced outer list
 *
 */
void HybridSetup::setupForcedOuterList()
{
    _engine.getSimulationBox().setupForcedOuterMolecules(
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
 * @throws customException::InputFileException if the layer radius exceeds one
 quarter of the smallest box dimension (minimum image convention)
 * @throws customException::InputFileException if the sum of layer radius and
 point charge thickness exceeds three quarters of the smallest box dimension
 (includes point charges from beyond immediate neighboring cells)
 */
void HybridSetup::checkZoneRadii()
{
    const auto coreRadius  = HybridSettings::getCoreRadius();
    const auto layerRadius = HybridSettings::getLayerRadius();
    const auto smoothingRegionThickness =
        HybridSettings::getSmoothingRegionThickness();
    const auto pointChargeThickness = HybridSettings::getPointChargeThickness();
    const auto minimalBoxDimension =
        _engine.getSimulationBox().getMinimalBoxDimension();

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

    if (layerRadius > (minimalBoxDimension / 4))
        throw(InputFileException(
            std::format(
                "Layer radius ({} Å) exceeds one quarter of the smallest box "
                "dimension ({} Å). This configuration is not allowed to ensure "
                "compliance with the minimum image convention.",
                layerRadius,
                minimalBoxDimension
            )
        ));

    if ((layerRadius + pointChargeThickness) > (minimalBoxDimension * 3 / 2))
        throw(InputFileException(
            std::format(
                "Layer radius ({} Å) plus point charge thickness ({} Å) "
                "exceeds three halves of the smallest box dimension ({} Å). "
                "This configuration is not allowed, as it would include point "
                "charges from beyond the immediate neighboring cells.",
                layerRadius,
                pointChargeThickness,
                minimalBoxDimension
            )
        ));
}