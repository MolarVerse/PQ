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

#ifndef _WATER_MODEL_SETTINGS_HPP_

#define _WATER_MODEL_SETTINGS_HPP_

#include <mstd/enum.hpp>

#include "enums/waterModels.hpp"
#include "setting.hpp"

/**
 * @class WaterModelSettings
 *
 * @brief static class to store settings of the water model
 *
 */
class WaterModelSettings
{
   private:
    Setting<WaterIntraModel> _waterIntraModel;
    Setting<WaterInterModel> _waterInterModel;

   public:
    /********************
     * standard getters *
     ********************/

    [[nodiscard]] bool            isWaterModelSet() const;
    [[nodiscard]] bool            isInterWaterModelSet() const;
    [[nodiscard]] WaterIntraModel getWaterIntraModel() const;
    [[nodiscard]] WaterInterModel getWaterInterModel() const;

    /********************
     * standard setters *
     ********************/

    void setWaterIntraModel(WaterIntraModel model);
    void setWaterInterModel(WaterInterModel model);
};

#endif   // _WATER_MODEL_SETTINGS_HPP_
