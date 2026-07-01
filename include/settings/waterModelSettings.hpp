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
#include <string>        // for string
#include <string_view>   // for string_view

namespace settings
{
    /**
     * @brief Enum for intramolecular water model types
     */
    enum class WaterIntraModel : size_t
    {
        NONE,
        SPC,
        SPC_E,
        SPC_FW,
        QSPC_FW,
        SPC_DC,
        H2O_DC,
        TIP3P,
        OPC3,
        SPC_MTR,
        TIP3P_MTR
    };

    [[nodiscard]] std::string string(const WaterIntraModel &waterIntraModel);

    /**
     * @brief Enum for intermolecular water model types
     */
    enum class WaterInterModel : size_t
    {
        NONE,
        SPC,
        SPC_E,
        SPC_FW,
        QSPC_FW,
        SPC_DC,
        H2O_DC,
        TIP3P,
        OPC3,
        SPC_MTR,
        TIP3P_MTR
    };

    [[nodiscard]] std::string string(const WaterInterModel &waterInterModel);

    /**
     * @class WaterModelSettings
     *
     * @brief static class to store settings of the water model
     *
     */
    class WaterModelSettings
    {
       private:
        static inline bool            _isWaterModelSet      = false;
        static inline bool            _isInterWaterModelSet = false;
        static inline WaterIntraModel _waterIntraModel = WaterIntraModel::NONE;
        static inline WaterInterModel _waterInterModel = WaterInterModel::NONE;

       public:
        WaterModelSettings()  = delete;
        ~WaterModelSettings() = delete;

        WaterModelSettings(const WaterModelSettings &)            = delete;
        WaterModelSettings(WaterModelSettings &&)                 = delete;
        WaterModelSettings &operator=(const WaterModelSettings &) = delete;
        WaterModelSettings &operator=(WaterModelSettings &&)      = delete;

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static bool            isWaterModelSet();
        [[nodiscard]] static bool            isInterWaterModelSet();
        [[nodiscard]] static WaterIntraModel getWaterIntraModel();
        [[nodiscard]] static WaterInterModel getWaterInterModel();

        /********************
         * standard setters *
         ********************/

        static void setIsWaterModelSet(const bool isSet);
        static void setIsInterWaterModelSet(const bool isSet);

        static void setWaterIntraModel(const std::string_view &model);
        static void setWaterIntraModel(const WaterIntraModel model);

        static void setWaterInterModel(const std::string_view &model);
        static void setWaterInterModel(const WaterInterModel model);
    };

}   // namespace settings

#endif   // _WATER_MODEL_SETTINGS_HPP_