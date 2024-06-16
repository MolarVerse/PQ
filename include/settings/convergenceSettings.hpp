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
        // clang-format off
        static inline std::optional<double> _energyConvergence;
        static inline std::optional<double> _relEnergyConvergence;
        static inline std::optional<double> _absEnergyConvergence;

        static inline std::optional<double> _forceConvergence;
        static inline std::optional<double> _relForceConvergence;
        static inline std::optional<double> _absForceConvergence;

        static inline std::optional<double> _maxForceConvergence;
        static inline std::optional<double> _relMaxForceConvergence;
        static inline std::optional<double> _absMaxForceConvergence;

        static inline std::optional<double> _rmsForceConvergence;
        static inline std::optional<double> _relRMSForceConvergence;
        static inline std::optional<double> _absRMSForceConvergence;

        static inline bool _useEnergyConvergence   = true;
        static inline bool _useForceConvergence    = true;
        static inline bool _useMaxForceConvergence = true;
        static inline bool _useRMSForceConvergence = true;

        static inline ConvStrategy _convergenceStrategy       = ConvStrategy::RIGOROUS;
        static inline ConvStrategy _energyConvergenceStrategy = ConvStrategy::RIGOROUS;
        static inline ConvStrategy _forceConvergenceStrategy  = ConvStrategy::RIGOROUS;
        // clang-format on

       public:
        [[nodiscard]] static ConvStrategy determineConvStrategy(
            const std::string_view &strategy
        );

        /***************************
         * standard setter methods *
         ***************************/

        static void setEnergyConvergence(const double);
        static void setRelEnergyConvergence(const double);
        static void setAbsEnergyConvergence(const double);

        static void setForceConvergence(const double);
        static void setRelForceConvergence(const double);
        static void setAbsForceConvergence(const double);

        static void setMaxForceConvergence(const double);
        static void setRelMaxForceConvergence(const double);
        static void setAbsMaxForceConvergence(const double);

        static void setRMSForceConvergence(const double);
        static void setRelRMSForceConvergence(const double);
        static void setAbsRMSForceConvergence(const double);

        static void setUseEnergyConvergence(const bool);
        static void setUseForceConvergence(const bool);
        static void setUseMaxForceConvergence(const bool);
        static void setUseRMSForceConvergence(const bool);

        static void setConvergenceStrategy(const ConvStrategy);
        static void setConvergenceStrategy(const std::string_view &);

        static void setEnergyConvergenceStrategy(const ConvStrategy);
        static void setEnergyConvergenceStrategy(const std::string_view &);

        static void setForceConvergenceStrategy(const ConvStrategy);
        static void setForceConvergenceStrategy(const std::string_view &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static std::optional<double> getEnergyConvergence();
        [[nodiscard]] static std::optional<double> getRelEnergyConvergence();
        [[nodiscard]] static std::optional<double> getAbsEnergyConvergence();

        [[nodiscard]] static std::optional<double> getForceConvergence();
        [[nodiscard]] static std::optional<double> getRelForceConvergence();
        [[nodiscard]] static std::optional<double> getAbsForceConvergence();

        [[nodiscard]] static std::optional<double> getMaxForceConvergence();
        [[nodiscard]] static std::optional<double> getRelMaxForceConvergence();
        [[nodiscard]] static std::optional<double> getAbsMaxForceConvergence();

        [[nodiscard]] static std::optional<double> getRMSForceConvergence();
        [[nodiscard]] static std::optional<double> getRelRMSForceConvergence();
        [[nodiscard]] static std::optional<double> getAbsRMSForceConvergence();

        [[nodiscard]] static bool getUseEnergyConvergence();
        [[nodiscard]] static bool getUseForceConvergence();
        [[nodiscard]] static bool getUseMaxForceConvergence();
        [[nodiscard]] static bool getUseRMSForceConvergence();

        [[nodiscard]] static ConvStrategy getConvergenceStrategy();
        [[nodiscard]] static ConvStrategy getEnergyConvergenceStrategy();
        [[nodiscard]] static ConvStrategy getForceConvergenceStrategy();
    };

}   // namespace settings

#endif   // _CONVERGENCE_SETTINGS_HPP_