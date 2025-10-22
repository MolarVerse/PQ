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

#include "exceptions.hpp"        // for customException
#include "stringUtilities.hpp"   // for toLowerCopy

using namespace settings;
using namespace utilities;
using namespace customException;

/********************
 * standard getters *
 ********************/

/**
 * @brief returns the waterIntraModel
 *
 * @return waterIntraModel
 */
WaterIntraModel WaterModelSettings::getWaterIntraModel()
{
    return _waterIntraModel;
}

/**
 * @brief returns the waterInterModel
 *
 * @return waterInterModel
 */
WaterInterModel WaterModelSettings::getWaterInterModel()
{
    return _waterInterModel;
}

/********************
 * standard setters *
 ********************/

/**
 * @brief sets the waterIntraModel to enum in settings
 *
 * @param model
 */
void WaterModelSettings::setWaterIntraModel(const std::string_view &model)
{
    using enum WaterIntraModel;
    const auto waterModel = toLowerAndReplaceDashesCopy(model);

    if ("spc/fw" == waterModel)
        _waterIntraModel = SPC_FW;
    else
        throw UserInputException(
            std::format("Water intra model \"{}\" not recognized", model)
        );
}

/**
 * @brief sets the waterIntraModel to enum in settings
 *
 * @param model
 */
void WaterModelSettings::setWaterIntraModel(const WaterIntraModel model)
{
    _waterIntraModel = model;
}

/**
 * @brief sets the waterInterModel to enum in settings
 *
 * @param model
 */
void WaterModelSettings::setWaterInterModel(const std::string_view &model)
{
    using enum WaterInterModel;
    const auto waterModel = toLowerAndReplaceDashesCopy(model);

    if ("spc/fw" == waterModel)
        _waterInterModel = SPC_FW;
    else
        throw UserInputException(
            std::format("Water inter model \"{}\" not recognized", model)
        );
}

/**
 * @brief sets the waterInterModel to enum in settings
 *
 * @param model
 */
void WaterModelSettings::setWaterInterModel(const WaterInterModel model)
{
    _waterInterModel = model;
}
