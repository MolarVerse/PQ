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

#include <format>

#include "exceptions.hpp"        // for customException
#include "stringUtilities.hpp"   // for toLowerCopy

using namespace settings;
using namespace utilities;
using namespace exc;

/********************
 * standard getters *
 ********************/

/**
 * @brief returns whether a water model is set
 *
 * @return true if water model is set
 */
bool WaterModelSettings::isWaterModelSet() { return _isWaterModelSet; }

/**
 * @brief returns whether a intermolecular water model is set
 *
 * @return true if intermolecular water model is set
 */
bool WaterModelSettings::isInterWaterModelSet()
{
    return _isInterWaterModelSet;
}

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
 * @brief sets whether a water model is set
 *
 * @param isSet
 */
void WaterModelSettings::setIsWaterModelSet(const bool isSet)
{
    _isWaterModelSet = isSet;
}

/**
 * @brief sets whether a intermolecular water model is set
 *
 * @param isSet
 */
void WaterModelSettings::setIsInterWaterModelSet(const bool isSet)
{
    _isInterWaterModelSet = isSet;
}

/**
 * @brief sets the waterIntraModel to enum in settings
 *
 * @param model
 */
void WaterModelSettings::setWaterIntraModel(const std::string_view &model)
{
    using enum WaterIntraModel;
    const auto waterModel = toLowerAndReplaceDashesCopy(model);

    if ("spc_e" == waterModel)
        _waterIntraModel = SPC_E;
    else if ("spc_fw" == waterModel)
        _waterIntraModel = SPC_FW;
    else if ("qspc_fw" == waterModel)
        _waterIntraModel = QSPC_FW;
    else if ("spc_dc" == waterModel)
        _waterIntraModel = SPC_DC;
    else if ("h2o_dc" == waterModel)
        _waterIntraModel = H2O_DC;
    else if ("tip3p" == waterModel)
        _waterIntraModel = TIP3P;
    else if ("opc3" == waterModel)
        _waterIntraModel = OPC3;
    else if ("spc_mtr" == waterModel)
        _waterIntraModel = SPC_MTR;
    else if ("tip3p_mtr" == waterModel)
        _waterIntraModel = TIP3P_MTR;
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

    if ("spc" == waterModel)
        _waterInterModel = SPC;
    else if ("spc_e" == waterModel)
        _waterInterModel = SPC_E;
    else if ("spc_fw" == waterModel)
        _waterInterModel = SPC_FW;
    else if ("qspc_fw" == waterModel)
        _waterInterModel = QSPC_FW;
    else if ("spc_dc" == waterModel)
        _waterInterModel = SPC_DC;
    else if ("h2o_dc" == waterModel)
        _waterInterModel = H2O_DC;
    else if ("tip3p" == waterModel)
        _waterInterModel = TIP3P;
    else if ("opc3" == waterModel)
        _waterInterModel = OPC3;
    else if ("spc_mtr" == waterModel)
        _waterInterModel = SPC_MTR;
    else if ("tip3p_mtr" == waterModel)
        _waterInterModel = TIP3P_MTR;
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

/**
 * @brief Convert a water intramolecular model enum to its string
 * representation.
 *
 * @param waterIntraModel The water intramolecular model enum value.
 *
 * @return A human-readable string for the model (e.g., "SPC/Fw", "TIP3P"),
 * or "none" if the model is unknown.
 */
std::string settings::string(const WaterIntraModel &waterIntraModel)
{
    switch (waterIntraModel)
    {
        using enum WaterIntraModel;

        case SPC: return "SPC";
        case SPC_E: return "SPC/E";
        case SPC_FW: return "SPC/Fw";
        case QSPC_FW: return "qSPC/Fw";
        case SPC_DC: return "SPC/DC";
        case H2O_DC: return "H2O-DC";
        case TIP3P: return "TIP3P";
        case OPC3: return "OPC3";
        case SPC_MTR: return "SPC-mTR";
        case TIP3P_MTR: return "TIP3P-mTR";
        case NONE: return "none";
    }
}

/**
 * @brief Convert a water intermolecular model enum to its string
 * representation.
 *
 * @param waterInterModel The water intermolecular model enum value.
 *
 * @return A human-readable string for the model (e.g., "SPC/Fw", "TIP3P"),
 * or "none" if the model is unknown.
 */
std::string settings::string(const WaterInterModel &waterInterModel)
{
    switch (waterInterModel)
    {
        using enum WaterInterModel;

        case SPC: return "SPC";
        case SPC_E: return "SPC/E";
        case SPC_FW: return "SPC/Fw";
        case QSPC_FW: return "qSPC/Fw";
        case SPC_DC: return "SPC/DC";
        case H2O_DC: return "H2O-DC";
        case TIP3P: return "TIP3P";
        case OPC3: return "OPC3";
        case SPC_MTR: return "SPC-mTR";
        case TIP3P_MTR: return "TIP3P-mTR";
        case NONE: return "none";
    }
}
