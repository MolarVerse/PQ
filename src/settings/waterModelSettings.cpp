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

#include "referenceSettings.hpp"
#include "rigidWaterGeometry.hpp"

/**
 * @brief Finalize the water model settings by adding references
 *
 * @param references Reference settings to be updated with water model
 * references
 */
void WaterModelSettings::finalize(ReferenceSettings &references)
{
    _addReferences(references);
}

/**
 * @brief Get the rigid geometry parameters for a water model.
 *
 * @details Only rigid water variants return geometry data. Flexible water
 * models return @c std::nullopt and are handled without constraints.
 *
 * @return The O-H and H-H distances for rigid models, or @c std::nullopt if
 * the model is not rigid.
 */
std::optional<RigidWaterGeometry> WaterModelSettings::getRigidWaterGeometry(
) const
{
    using enum WaterIntraModel;

    if (!_waterIntraModel.isSet())
        return std::nullopt;

    switch (_waterIntraModel.get())
    {
        case SPC:
            return RigidWaterGeometry{.dOH = SPC_OH_DIST, .dHH = SPC_HH_DIST};
        case SPC_E:
            return RigidWaterGeometry{
                .dOH = SPC_E_OH_DIST,
                .dHH = SPC_E_HH_DIST
            };
        case SPC_DC:
            return RigidWaterGeometry{
                .dOH = SPC_DC_OH_DIST,
                .dHH = SPC_DC_HH_DIST
            };
        case H2O_DC:
            return RigidWaterGeometry{
                .dOH = H2O_DC_OH_DIST,
                .dHH = H2O_DC_HH_DIST
            };
        case TIP3P:
            return RigidWaterGeometry{
                .dOH = TIP3P_OH_DIST,
                .dHH = TIP3P_HH_DIST
            };
        case OPC3:
            return RigidWaterGeometry{.dOH = OPC3_OH_DIST, .dHH = OPC3_HH_DIST};
        case SPC_FW:
        case QSPC_FW:
        case SPC_MTR:
        case TIP3P_MTR:
        case NONE: return std::nullopt;
    }

    std::unreachable();
}

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

/**
 * @brief Add reference file entries for the configured water models.
 *
 * @details Adds bibliography references for both intramolecular and
 * intermolecular water models to the references output.
 */
void WaterModelSettings::_addReferences(ReferenceSettings &references) const
{
    if (_waterIntraModel.isSet())
    {
        switch (_waterIntraModel.get())
        {
            using enum WaterIntraModel;
            case SPC:
                references.addReferenceFile(ReferenceFiles::SPC_FILE);
                break;
            case SPC_E:
                references.addReferenceFile(ReferenceFiles::SPC_E_FILE);
                break;
            case SPC_FW:
                references.addReferenceFile(ReferenceFiles::SPC_FW_FILE);
                break;
            case QSPC_FW:
                references.addReferenceFile(ReferenceFiles::QSPC_FW_FILE);
                break;
            case SPC_DC:
                references.addReferenceFile(ReferenceFiles::SPC_DC_FILE);
                break;
            case H2O_DC:
                references.addReferenceFile(ReferenceFiles::H2O_DC_FILE);
                break;
            case TIP3P:
                references.addReferenceFile(ReferenceFiles::TIP3P_FILE);
                break;
            case OPC3:
                references.addReferenceFile(ReferenceFiles::OPC3_FILE);
                break;
            case SPC_MTR:
                references.addReferenceFile(ReferenceFiles::SPC_MTR_FILE);
                break;
            case TIP3P_MTR:
                references.addReferenceFile(ReferenceFiles::TIP3P_MTR_FILE);
                break;
            case NONE: break;
        }
    }

    if (_waterInterModel.isSet())
    {
        switch (_waterInterModel.get())
        {
            using enum WaterInterModel;
            case SPC:
                references.addReferenceFile(ReferenceFiles::SPC_FILE);
                break;
            case SPC_E:
                references.addReferenceFile(ReferenceFiles::SPC_E_FILE);
                break;
            case SPC_FW:
                references.addReferenceFile(ReferenceFiles::SPC_FW_FILE);
                break;
            case QSPC_FW:
                references.addReferenceFile(ReferenceFiles::QSPC_FW_FILE);
                break;
            case SPC_DC:
                references.addReferenceFile(ReferenceFiles::SPC_DC_FILE);
                break;
            case H2O_DC:
                references.addReferenceFile(ReferenceFiles::H2O_DC_FILE);
                break;
            case TIP3P:
                references.addReferenceFile(ReferenceFiles::TIP3P_FILE);
                break;
            case OPC3:
                references.addReferenceFile(ReferenceFiles::OPC3_FILE);
                break;
            case SPC_MTR:
                references.addReferenceFile(ReferenceFiles::SPC_MTR_FILE);
                break;
            case TIP3P_MTR:
                references.addReferenceFile(ReferenceFiles::TIP3P_MTR_FILE);
                break;
            case NONE: break;
        }
    }
}
