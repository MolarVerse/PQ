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

#include "hybridSettings.hpp"

using settings::HybridSettings;

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief set the innerRegionCenter string in the settings
 *
 * @details the innerRegionCenter string is a string representation of a
 * selection with which the center of the inner region of a hybrid calculation
 * can be selected
 *
 * @param innerRegionCenter
 */
void HybridSettings::setInnerRegionCenterString(
    const std::string_view &innerRegionCenter
)
{
    _innerRegionCenterString = innerRegionCenter;
}

/**
 * @brief set the _forcedInnerListString
 *
 * @details the forcedInnterListString is a string representation of a selection
 * with which the atoms of the inner region of the hybrid calculation can be
 * selected
 *
 * @param list
 */
void HybridSettings::setForcedInnerListString(const std::string_view &list)
{
    _forcedInnerListString = list;
}

/**
 * @brief set the _forcedOuterListString
 *
 * @details the forcedOuterListString is a string representation of a selection
 * with which the atoms of the outer region of the hybrid calculation can be
 * selected
 *
 * @param list
 */
void HybridSettings::setForcedOuterListString(
    const std::string_view &forcedOuterList
)
{
    _forcedOuterListString = forcedOuterList;
}

/**
 * @brief set the useQMCharges in the settings
 *
 * @param useQMCharges
 */
void HybridSettings::setUseQMCharges(const bool useQMCharges)
{
    _useQMCharges = useQMCharges;
}

/**
 * @brief set the coreRadius in the settings
 *
 * @details the coreRadius is the radius of the core region
 *
 * @param qmCoreRadius
 */
void HybridSettings::setCoreRadius(const double radius)
{
    _coreRadius = radius;
}

/**
 * @brief set the layerRadius in the settings
 *
 * @details the layerRadius is the radius of the layer region
 *
 * @param qmmmLayerRadius
 */
void HybridSettings::setLayerRadius(const double radius)
{
    _layerRadius = radius;
}

/**
 * @brief set the smoothingRadius in the settings
 *
 * @details the smoothingRadius is the radius of the smoothing region
 *
 * @param qmmmSmoothingRadius
 */
void HybridSettings::setSmoothingRadius(const double radius)
{
    _smoothingRadius = radius;
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get the innerRegionCenter string
 *
 * @return std::string
 */
std::string HybridSettings::getInnerRegionCenterString()
{
    return _innerRegionCenterString;
}

/**
 * @brief get the coreOnlyList string
 *
 * @return std::string
 */
std::string HybridSettings::getForcedInnerListString()
{
    return _forcedInnerListString;
}

/**
 * @brief get the nonCoreOnlyList string
 *
 * @return std::string
 */
std::string HybridSettings::getForcedOuterListString()
{
    return _forcedOuterListString;
}

/**
 * @brief get the useQMCharges
 *
 * @return bool
 */
bool HybridSettings::getUseQMCharges() { return _useQMCharges; }

/**
 * @brief get the coreRadius
 *
 * @return double
 */
double HybridSettings::getCoreRadius() { return _coreRadius; }

/**
 * @brief get the layerRadius
 *
 * @return double
 */
double HybridSettings::getLayerRadius() { return _layerRadius; }

/**
 * @brief get the smoothingRadius
 *
 * @return double
 */
double HybridSettings::getSmoothingRadius() { return _smoothingRadius; }