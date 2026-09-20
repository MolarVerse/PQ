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

#ifndef _MANOSTAT_SETTINGS_HPP_

#define _MANOSTAT_SETTINGS_HPP_

#include <cstdint>
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "defaults.hpp"
#include "mstd/enum.hpp"

namespace settings
{
    /**
     * @enum ManostatType
     *
     * @brief enum class to store the type of the manostat
     *
     */
    enum class ManostatType : std::uint8_t
    {
        NONE,
        BERENDSEN,
        STOCHASTIC_RESCALING
    };

    /**
     * @enum Isotropy
     *
     * @brief enum class to store the isotropy of the manostat
     *
     */
    enum class Isotropy : std::uint8_t
    {
        NONE,
        ISOTROPIC,
        SEMI_ISOTROPIC,
        ANISOTROPIC,
        FULL_ANISOTROPIC
    };

    // clang-format off
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define FIXED_AXIS_LIST(axis) \
    axis(NONE, 0U)    \
    axis(X, 1U << 0U) \
    axis(Y, 1U << 1U) \
    axis(Z, 1U << 2U) \
    axis(XY, 0B011)   \
    axis(XZ, 0B101)   \
    axis(YZ, 0B110)   \
    axis(ALL, 0B111)
    // clang-format on

    MSTD_ENUM_BITFLAG(FixedAxis, std::uint8_t, FIXED_AXIS_LIST);

#undef FIXED_AXIS_LIST

    [[nodiscard]] constexpr bool isAxisFixed(
        FixedAxis fixedAxis,
        size_t    axisIndex
    )
    {
        const auto axisToCheck = static_cast<FixedAxis>(1U << axisIndex);
        return (fixedAxis & axisToCheck) == axisToCheck;
    }

    [[nodiscard]] std::string string(const ManostatType &manostatType);
    [[nodiscard]] std::string string(const Isotropy &isotropy);
    [[nodiscard]] std::string string(const FixedAxis &fixedAxis);

    /**
     * @class ManostatSettings
     *
     * @brief static class to store settings of the manostat
     *
     */
    class ManostatSettings
    {
       private:
        static inline ManostatType _manostatType   = ManostatType::NONE;
        static inline Isotropy     _isotropy       = Isotropy::ISOTROPIC;
        static inline FixedAxis    _fixedAxis      = FixedAxis::ALL;
        static inline bool         _isFixedAxisSet = false;

        static inline double _targetPressure;

        // clang-format off
        static inline double _tauManostat     = defaults::BERENDSEN_MANOSTAT_RELAX_TIME;
        static inline double _compressibility = defaults::COMPRESSIBILITY_WATER_DEFAULT;
        // clang-format on

        static inline std::vector<size_t> _2DIsotropicAxes;
        static inline size_t              _2DAnisotropicAxis;

       public:
        ManostatSettings()  = default;
        ~ManostatSettings() = default;

        /***************************
         * standard setter methods *
         ***************************/

        static void setManostatType(const std::string_view &manostatType);
        static void setManostatType(const ManostatType &manostatType);

        static void setIsotropy(const std::string_view &isotropy);
        static void setIsotropy(const Isotropy &isotropy);

        static void setFixedAxis(const FixedAxis &fixedAxis);
        static void setIsFixedAxisSet(bool isSet);

        static void setTargetPressure(double targetPressure);
        static void setTauManostat(double tauManostat);
        static void setCompressibility(double compressibility);

        static void set2DIsotropicAxes(const std::vector<size_t> &indices);
        static void set2DAnisotropicAxis(size_t index);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static bool isBerendsenBased();

        [[nodiscard]] static ManostatType        getManostatType();
        [[nodiscard]] static Isotropy            getIsotropy();
        [[nodiscard]] static FixedAxis           getFixedAxis();
        [[nodiscard]] static bool                isFixedAxisSet();
        [[nodiscard]] static double              getTargetPressure();
        [[nodiscard]] static double              getTauManostat();
        [[nodiscard]] static double              getCompressibility();
        [[nodiscard]] static std::vector<size_t> get2DIsotropicAxes();
        [[nodiscard]] static size_t              get2DAnisotropicAxis();
    };

}   // namespace settings

#endif   // _MANOSTAT_SETTINGS_HPP_
