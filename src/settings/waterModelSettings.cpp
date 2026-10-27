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

#include "waterModelSettings.hpp"

/********************
 * standard getters *
 ********************/

/**
 * @brief returns whether a water model is set
 *
 * @return true if water model is set
 */
bool WaterModelSettings::isWaterModelSet() const
{
    return _waterIntraModel.isSet() || _waterInterModel.isSet();
}

/**
 * @brief returns whether a intermolecular water model is set
 *
 * @return true if intermolecular water model is set
 */
bool WaterModelSettings::isInterWaterModelSet() const
{
    return _waterInterModel.isSet();
}

/**
 * @brief returns the waterIntraModel
 *
 * @return waterIntraModel
 */
WaterIntraModel WaterModelSettings::getWaterIntraModel() const
{
    return _waterIntraModel.get();
}

/**
 * @brief returns the waterInterModel
 *
 * @return waterInterModel
 */
WaterInterModel WaterModelSettings::getWaterInterModel() const
{
    return _waterInterModel.get();
}

/********************
 * standard setters *
 ********************/

/**
 * @brief sets the waterIntraModel to enum in settings
 *
 * @param model
 */
void WaterModelSettings::setWaterIntraModel(WaterIntraModel model)
{
    _waterIntraModel.set(model);
}

/**
 * @brief sets the waterInterModel to enum in settings
 *
 * @param model
 */
void WaterModelSettings::setWaterInterModel(WaterInterModel model)
{
    _waterInterModel.set(model);
}
