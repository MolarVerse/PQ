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

#include <vector>

using settings::HybridSettings;

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief set the innerRegionCenter in the settings
 *
 * @details the innerRegionCenter is a list of atom indices with which the
 * center of the inner region of a hybrid calculation can be selected
 *
 * @param innerRegionCenter
 */
void HybridSettings::setInnerRegionCenter(
    const std::vector<int> &innerRegionCenter
)
{
    _innerRegionCenter = innerRegionCenter;
}

/**
 * @brief set the _forcedInnerList in the settings
 *
 * @details the forcedInnerList is a list of atoms which will always be treated
 * with the method chosen for the inner region of the hybrid calculation
 *
 * @param list
 */
void HybridSettings::setForcedInnerList(const std::vector<int> &list)
{
    _forcedInnerList = list;
}

/**
 * @brief set the _forcedOuterList in the settings
 *
 * @details the forcedInnerList is a list of atoms which will always be treated
 * with the method chosen for the outer region of the hybrid calculation
 *
 * @param list
 */
void HybridSettings::setForcedOuterList(const std::vector<int> &list)
{
    _forcedOuterList = list;
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
 * @brief set the smoothingRegionThickness in the settings
 *
 * @param thickness
 */
void HybridSettings::setSmoothingRegionThickness(const double thickness)
{
    _smoothingRegionThickness = thickness;
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get the innerRegionCenter as list of int
 *
 * @return vector<int>
 */
std::vector<int> HybridSettings::getInnerRegionCenter()
{
    return _innerRegionCenter;
}

/**
 * @brief get the forcedInnerList
 *
 * @return vector<int>
 */
std::vector<int> HybridSettings::getForcedInnerList()
{
    return _forcedInnerList;
}

/**
 * @brief get the forcedOuterList
 *
 * @return vector<int>
 */
std::vector<int> HybridSettings::getForcedOuterList()
{
    return _forcedOuterList;
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
 * @brief get the smoothingRegionThickness
 *
 * @return double
 */
double HybridSettings::getSmoothingRegionThickness()
{
    return _smoothingRegionThickness;
}