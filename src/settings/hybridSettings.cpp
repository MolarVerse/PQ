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
#include <format>
#include <vector>

#include "exceptions.hpp"
#include "fileSettings.hpp"

#ifdef PYTHON_ENABLED
#include "selection.hpp"
#endif

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
void HybridSettings::setInnerRegionCenterByUser(
    const std::vector<size_t> &innerRegionCenter
)
{
    _innerRegionCenter.setByUser(innerRegionCenter);
}

/**
 * @brief set a pending selection expression for the innerRegionCenter
 *
 * @details the expression is resolved to atom indices in finalize(),
 * because the start and moldescriptor files might not be parsed yet
 *
 * @param expression
 */
void HybridSettings::setInnerRegionCenterExpression(
    const std::string &expression
)
{
    _innerRegionCenterExpression = expression;
}

/**
 * @brief resolve the pending innerRegionCenter selection expression
 *
 * @param fileSettings
 *
 * @throw exc::InputFileException if the selection is invalid or empty
 */
void HybridSettings::finalize(const FileSettings &fileSettings)
{
    if (!_innerRegionCenterExpression.has_value())
        return;

    const auto expression = *_innerRegionCenterExpression;
    _innerRegionCenterExpression.reset();

#ifdef PYTHON_ENABLED
    auto selection = pq_python::select(
        expression,
        fileSettings.getStartFileName(),
        fileSettings.getMolDescriptorFileName()
    );

    std::ranges::sort(selection);
    const auto [first, last] = std::ranges::unique(selection);
    selection.erase(first, last);

    if (selection.empty() || selection.front() < 0)
        throw exc::InputFileException(
            std::format(
                "Invalid selection \"{}\" for \"inner_region_center\": "
                "the selection is empty or contains negative atom indices",
                expression
            )
        );

    _innerRegionCenter.setByUser(
        std::vector<size_t>(selection.begin(), selection.end())
    );
#else
    static_cast<void>(fileSettings);
    throw exc::InputFileException(
        std::format(
            "Selection \"{}\" for \"inner_region_center\" requires "
            "PQ to be built with Python bindings",
            expression
        )
    );
#endif
}

/**
 * @brief set the _forcedCoreList in the settings
 *
 * @details the forcedCoreList is a list of molecules which will always be
 * part of the CORE zone in hybrid calculation
 *
 * @param list
 */
void HybridSettings::setForcedCoreListByUser(const std::vector<int> &list)
{
    _forcedCoreList.setByUser(list);
}

/**
 * @brief set the _forcedLayerList in the settings
 *
 * @details the forcedLayerList is a list of molecules which will always be
 * part of the LAYER zone in hybrid calculation
 *
 * @param list
 */
void HybridSettings::setForcedLayerListByUser(const std::vector<int> &list)
{
    _forcedLayerList.setByUser(list);
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
void HybridSettings::setForcedOuterListByUser(const std::vector<int> &list)
{
    _forcedOuterList.setByUser(list);
}

/**
 * @brief set the useQMCharges in the settings
 *
 * @param useQMCharges
 */
void HybridSettings::setUseQMChargesByUser(bool useQMCharges)
{
    _useQMCharges.setByUser(useQMCharges);
}

/**
 * @brief set the coreRadius in the settings
 *
 * @details the coreRadius is the radius of the core region
 *
 * @param radius
 */
void HybridSettings::setCoreRadiusByUser(const Distance &radius)
{
    _coreRadius.setByUser(radius);
}

/**
 * @brief set the layerRadius in the settings
 *
 * @details the layerRadius is the radius of the layer region
 *
 * @param radius
 */
void HybridSettings::setLayerRadiusByUser(const Distance &radius)
{
    _layerRadius.setByUser(radius);
}

/**
 * @brief set the smoothingRegionThickness in the settings
 *
 * @param thickness
 */
void HybridSettings::setSmoothingRegionThicknessByUser(
    const Distance &thickness
)
{
    _smoothingRegionThickness.setByUser(thickness);
}

/**
 * @brief set the pointChargeThickness in the settings
 *
 * @details the pointChargeThickness is the distance measured from the layer
 * radius up to which point charges are included
 *
 * @param radius
 */
void HybridSettings::setPointChargeThicknessByUser(const Distance &radius)
{
    _pointChargeThickness.setByUser(radius);
}

/**
 * @brief set the smoothing method in the settings
 *
 * @param method
 */
void HybridSettings::setSmoothingMethodByUser(SmoothingMethod method)
{
    _smoothing.setByUser(method);
}

/**
 * @brief set the type of QM force distribution in hotspot smoothing
 *
 * @param method
 */
void HybridSettings::setQMForceDistByUser(QMForceDist method)
{
    _qmForceDist.setByUser(method);
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get the innerRegionCenter as list of int
 *
 * @return const std::vector<size_t> &
 */
const std::vector<size_t> &HybridSettings::getInnerRegionCenter() const
{
    return _innerRegionCenter.get();
}

/**
 * @brief get the forcedCoreList
 *
 * @return const vector<int>&
 */
const std::vector<int> &HybridSettings::getForcedCoreList() const
{
    return _forcedCoreList.get();
}

/**
 * @brief get the forcedLayerList
 *
 * @return const vector<int>&
 */
const std::vector<int> &HybridSettings::getForcedLayerList() const
{
    return _forcedLayerList.get();
}

/**
 * @brief get the forcedOuterList
 *
 * @return const vector<int>&
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
 * @return const Distance &
 */
const Distance &HybridSettings::getCoreRadius() const
{
    return _coreRadius.get();
}

/**
 * @brief get the layerRadius
 *
 * @return const Distance &
 */
const Distance &HybridSettings::getLayerRadius() const
{
    return _layerRadius.get();
}

/**
 * @brief get the smoothingRegionThickness
 *
 * @return const Distance &
 */
const Distance &HybridSettings::getSmoothingRegionThickness() const
{
    return _smoothingRegionThickness.get();
}

/**
 * @brief get the pointChargeThickness
 *
 * @return const Distance &
 */
const Distance &HybridSettings::getPointChargeThickness() const
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
