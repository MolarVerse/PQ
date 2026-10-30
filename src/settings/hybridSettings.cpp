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

#include <cstddef>
#include <vector>

/**
 * @brief Construct a new Hybrid Settings:: Hybrid Settings object
 *
 */
HybridSettings::HybridSettings()
    : _forcedCoreList({}),
      _forcedLayerList({}),
      _forcedOuterList({}),
      _useQMCharges(true),
      _coreRadius(0.0),
      _layerRadius(0.0),
      _smoothingRegionThickness(0.0),
      _pointChargeThickness(0.0),
      _smoothing(SmoothingMethod::HOTSPOT),
      _qmForceDist(QMForceDist::NONE)
{
}

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
    const std::vector<size_t> &innerRegionCenter
)
{
    _innerRegionCenter.set(innerRegionCenter);
}

/**
 * @brief set the _forcedCoreList in the settings
 *
 * @details the forcedCoreList is a list of molecules which will always be
 * part of the CORE zone in hybrid calculation
 *
 * @param list
 */
void HybridSettings::setForcedCoreList(const std::vector<int> &list)
{
    _forcedCoreList.set(list);
}

/**
 * @brief set the _forcedLayerList in the settings
 *
 * @details the forcedLayerList is a list of molecules which will always be
 * part of the LAYER zone in hybrid calculation
 *
 * @param list
 */
void HybridSettings::setForcedLayerList(const std::vector<int> &list)
{
    _forcedLayerList.set(list);
}

/**
 * @brief set the _forcedOuterList in the settings
 *
 * @details the forcedOuterList is a list of molecules which will always be
 * treated with the method chosen for the outer region of the hybrid
 * calculation
 *
 * @param list
 */
void HybridSettings::setForcedOuterList(const std::vector<int> &list)
{
    _forcedOuterList.set(list);
}

/**
 * @brief set the useQMCharges in the settings
 *
 * @param useQMCharges
 */
void HybridSettings::setUseQMCharges(bool useQMCharges)
{
    _useQMCharges.set(useQMCharges);
}

/**
 * @brief set the coreRadius in the settings
 *
 * @details the coreRadius is the radius of the core region
 *
 * @param radius
 */
void HybridSettings::setCoreRadius(double radius) { _coreRadius.set(radius); }

/**
 * @brief set the layerRadius in the settings
 *
 * @details the layerRadius is the radius of the layer region
 *
 * @param radius
 */
void HybridSettings::setLayerRadius(double radius) { _layerRadius.set(radius); }

/**
 * @brief set the smoothingRegionThickness in the settings
 *
 * @param thickness
 */
void HybridSettings::setSmoothingRegionThickness(double thickness)
{
    _smoothingRegionThickness.set(thickness);
}

/**
 * @brief set the pointChargeThickness in the settings
 *
 * @details the pointChargeThickness is the distance measured from the layer
 * radius up to which point charges are included
 *
 * @param radius
 */
void HybridSettings::setPointChargeThickness(double radius)
{
    _pointChargeThickness.set(radius);
}

/**
 * @brief set the smoothing method in the settings
 *
 * @param method
 */
void HybridSettings::setSmoothingMethod(SmoothingMethod method)
{
    _smoothing.set(method);
}

/**
 * @brief set the type of QM force distribtion in hotspot smoothing
 *
 * @param method
 */
void HybridSettings::setQMForceDist(QMForceDist method)
{
    _qmForceDist.set(method);
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get the innerRegionCenter as list of int
 *
 * @return std::optional<std::vector<size_t>>
 */
std::optional<std::vector<size_t>> HybridSettings::getInnerRegionCenter() const
{
    if (_innerRegionCenter.isSet())
        return _innerRegionCenter.get();

    return std::nullopt;
}

/**
 * @brief get the forcedCoreList
 *
 * @return vector<int>
 */
const std::vector<int> &HybridSettings::getForcedCoreList() const
{
    return _forcedCoreList.get();
}

/**
 * @brief get the forcedLayerList
 *
 * @return vector<int>
 */
const std::vector<int> &HybridSettings::getForcedLayerList() const
{
    return _forcedLayerList.get();
}

/**
 * @brief get the forcedOuterList
 *
 * @return vector<int>
 */
const std::vector<int> &HybridSettings::getForcedOuterList() const
{
    return _forcedOuterList.get();
}

/**
 * @brief get the useQMCharges
 *
 * @return bool
 */
bool HybridSettings::getUseQMCharges() const { return _useQMCharges.get(); }

/**
 * @brief get the coreRadius
 *
 * @return double
 */
double HybridSettings::getCoreRadius() const { return _coreRadius.get(); }

/**
 * @brief get the layerRadius
 *
 * @return double
 */
double HybridSettings::getLayerRadius() const { return _layerRadius.get(); }

/**
 * @brief get the smoothingRegionThickness
 *
 * @return double
 */
double HybridSettings::getSmoothingRegionThickness() const
{
    return _smoothingRegionThickness.get();
}

/**
 * @brief get the pointChargeThickness
 *
 * @return double
 */
double HybridSettings::getPointChargeThickness() const
{
    return _pointChargeThickness.get();
}

/**
 * @brief get the smoothing method
 *
 * @return SmoothingMethod
 */
SmoothingMethod HybridSettings::getSmoothingMethod() const
{
    return _smoothing.get();
}

/**
 * @brief get the type of QM force distribution in hotspot smoothing
 *
 * @return QMForceDist
 */
QMForceDist HybridSettings::getQMForceDist() const
{
    return _qmForceDist.get();
}
