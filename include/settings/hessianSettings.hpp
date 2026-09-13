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

#ifndef _HESSIAN_SETTINGS_HPP_

#define _HESSIAN_SETTINGS_HPP_

#include <cstdint>
#include <string>
#include <string_view>

#include "defaults.hpp"

namespace settings
{
    enum class HessianBuilderType : std::uint8_t
    {
        FINITE_DIFFERENCE_FORCES_CENTRAL,
        FINITE_DIFFERENCE_FORCES_FORWARD,
        FINITE_DIFFERENCE_FORCES_FIVE_POINT,
        ANALYTIC,
        NONE
    };

    [[nodiscard]]
    std::string string(HessianBuilderType builder);

    class HessianSettings
    {
       private:
        static inline std::string _hessianFile = DefaultFiles::hessianFile;
        static inline std::string _hessianInfoFile =
            DefaultFiles::hessianInfoFile;
        static inline double _displacement =
            defaults::HESSIAN_DISPLACEMENT_DEFAULT;
        static inline bool _optimizeBeforeHessian =
            defaults::HESSIAN_OPTIMIZE_DEFAULT;
        static inline HessianBuilderType _builder =
            HessianBuilderType::FINITE_DIFFERENCE_FORCES_CENTRAL;

       public:
        static void setHessianFile(const std::string_view &filename);
        static void setHessianInfoFile(const std::string_view &filename);
        static void setDisplacement(double displacement);
        static void setOptimizeBeforeHessian(bool optimize);
        static void setBuilder(const std::string_view &builder);
        static void setBuilder(HessianBuilderType builder);

        [[nodiscard]] static std::string        getHessianFile();
        [[nodiscard]] static std::string        getHessianInfoFile();
        [[nodiscard]] static double             getDisplacement();
        [[nodiscard]] static bool               optimizeBeforeHessian();
        [[nodiscard]] static HessianBuilderType getBuilder();
    };

}   // namespace settings

#endif   // _HESSIAN_SETTINGS_HPP_
