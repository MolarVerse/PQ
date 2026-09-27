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

#include <cstdint>
#include <mstd/enum.hpp>
#include <string_view>

namespace settings
{

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define WATER_INTRA_MODEL_LIST(X) \
    X(NONE)                       \
    X(SPC)                        \
    X(SPC_E)                      \
    X(SPC_FW)                     \
    X(QSPC_FW)                    \
    X(SPC_DC)                     \
    X(H2O_DC)                     \
    X(TIP3P)                      \
    X(OPC3)                       \
    X(SPC_MTR)                    \
    X(TIP3P_MTR)

    MSTD_ENUM(WaterIntraModel, std::uint8_t, WATER_INTRA_MODEL_LIST)

#undef WATER_INTRA_MODEL_LIST

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define WATER_INTER_MODEL_LIST(X) \
    X(NONE)                       \
    X(SPC)                        \
    X(SPC_E)                      \
    X(SPC_FW)                     \
    X(QSPC_FW)                    \
    X(SPC_DC)                     \
    X(H2O_DC)                     \
    X(TIP3P)                      \
    X(OPC3)                       \
    X(SPC_MTR)                    \
    X(TIP3P_MTR)

    MSTD_ENUM(WaterInterModel, std::uint8_t, WATER_INTER_MODEL_LIST)

#undef WATER_INTER_MODEL_LIST

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

        static void setIsWaterModelSet(bool isSet);
        static void setIsInterWaterModelSet(bool isSet);

        static void setWaterIntraModel(const std::string_view &model);
        static void setWaterIntraModel(WaterIntraModel model);

        static void setWaterInterModel(const std::string_view &model);
        static void setWaterInterModel(WaterInterModel model);
    };

}   // namespace settings

#endif   // _WATER_MODEL_SETTINGS_HPP_
