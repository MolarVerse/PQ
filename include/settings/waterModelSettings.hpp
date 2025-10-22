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

#include <cstddef>       // for size_t
#include <string_view>   // for string_view

namespace settings
{
    /**
     * @brief Enum for intramolecular water model types
     */
    enum class WaterIntraModel : size_t
    {
        NONE,
        SPC_FW
    };

    /**
     * @brief Enum for intermolecular water model types
     */
    enum class WaterInterModel : size_t
    {
        NONE,
        SPC_FW
    };

    /**
     * @class WaterModelSettings
     *
     * @brief static class to store settings of the water model
     *
     */
    class WaterModelSettings
    {
       private:
        static inline WaterIntraModel _waterIntraModel = WaterIntraModel::NONE;
        static inline WaterInterModel _waterInterModel = WaterInterModel::NONE;

       public:
        WaterModelSettings()  = default;
        ~WaterModelSettings() = default;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static WaterIntraModel getWaterIntraModel();
        [[nodiscard]] static WaterInterModel getWaterInterModel();

        /********************
         * standard setters *
         ********************/

        static void setWaterIntraModel(const std::string_view &model);
        static void setWaterIntraModel(const WaterIntraModel model);

        static void setWaterInterModel(const std::string_view &model);
        static void setWaterInterModel(const WaterInterModel model);
    };

}   // namespace settings

#endif   // _WATER_MODEL_SETTINGS_HPP_