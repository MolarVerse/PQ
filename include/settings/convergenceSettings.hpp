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

#ifndef _CONVERGENCE_SETTINGS_HPP_

#define _CONVERGENCE_SETTINGS_HPP_

#include <cstddef>       // for size_t
#include <optional>      // for optional
#include <string>        // for string
#include <string_view>   // for string_view

#include "defaults.hpp"   // for _OPTIMIZER_DEFAULT_

namespace settings
{
    /**
     * @brief enum ConvStrategy
     *
     */
    enum class ConvStrategy : size_t
    {
        RIGOROUS,
        LOOSE,
        ABSOLUTE,
        RELATIVE
    };

    std::string string(const ConvStrategy method);

    /**
     * @class ConvSettings
     *
     * @brief stores all information about the optimizer
     *
     */
    class ConvSettings
    {
       private:
        static inline std::optional<double> _energyConv;
        static inline std::optional<double> _relEnergyConv;
        static inline std::optional<double> _absEnergyConv;

        static inline std::optional<double> _forceConv;
        static inline std::optional<double> _maxForceConv;
        static inline std::optional<double> _rmsForceConv;

        static inline bool _useEnergyConv   = true;
        static inline bool _useForceConv    = true;
        static inline bool _useMaxForceConv = true;
        static inline bool _useRMSForceConv = true;

        static inline std::optional<ConvStrategy> _energyConvStrategy;

        // clang-format off
        static inline std::string _defaultEnergyConvStrategy = defaults::_EN_CONV_STRATEGY_DEFAULT_;
        // clang-format on

       public:
        [[nodiscard]] static ConvStrategy getConvStrategy(
            const std::string_view &strategy
        );

        /***************************
         * standard setter methods *
         ***************************/

        static void setEnergyConv(const double);
        static void setRelEnergyConv(const double);
        static void setAbsEnergyConv(const double);

        static void setForceConv(const double);
        static void setMaxForceConv(const double);
        static void setRMSForceConv(const double);

        static void setUseEnergyConv(const bool);
        static void setUseForceConv(const bool);
        static void setUseMaxForceConv(const bool);
        static void setUseRMSForceConv(const bool);

        static void setEnergyConvStrategy(const ConvStrategy);
        static void setEnergyConvStrategy(const std::string_view &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static std::optional<double> getEnergyConv();
        [[nodiscard]] static std::optional<double> getRelEnergyConv();
        [[nodiscard]] static std::optional<double> getAbsEnergyConv();

        [[nodiscard]] static std::optional<double> getForceConv();
        [[nodiscard]] static std::optional<double> getMaxForceConv();
        [[nodiscard]] static std::optional<double> getRMSForceConv();

        [[nodiscard]] static bool getUseEnergyConv();
        [[nodiscard]] static bool getUseForceConv();
        [[nodiscard]] static bool getUseMaxForceConv();
        [[nodiscard]] static bool getUseRMSForceConv();

        [[nodiscard]] static std::optional<ConvStrategy> getEnConvStrategy();
        [[nodiscard]] static ConvStrategy getDefaultEnergyConvStrategy();
    };

}   // namespace settings

#endif   // _CONVERGENCE_SETTINGS_HPP_