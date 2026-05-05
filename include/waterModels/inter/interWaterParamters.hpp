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

#ifndef _INTER_WATER_PARAMETERS_TPP_HPP_

#define _INTER_WATER_PARAMETERS_TPP_HPP_

#include <concepts>
#include <vector>

#include "guffCoefficients.hpp"   // for Guff coefficients
#include "interWater.hpp"         // for InterWater

namespace waterModel
{
    template <class T>
    concept InterWaterParameterClass = requires {
        { T::_oxygenCharge } -> std::convertible_to<double>;
        { T::_hydrogenCharge } -> std::convertible_to<double>;
        { T::_guffCoefficientsOO } -> std::convertible_to<std::vector<double>>;
        { T::_guffCoefficientsOH } -> std::convertible_to<std::vector<double>>;
        { T::_guffCoefficientsHH } -> std::convertible_to<std::vector<double>>;
    };

    template <InterWaterParameterClass T>
    inline InterWaterState makeInterWaterState()
    {
        auto state                = InterWaterState();
        state._oxygenCharge       = T::_oxygenCharge;
        state._hydrogenCharge     = T::_hydrogenCharge;
        state._guffCoefficientsOO = T::_guffCoefficientsOO;
        state._guffCoefficientsOH = T::_guffCoefficientsOH;
        state._guffCoefficientsHH = T::_guffCoefficientsHH;

        return state;
    }

    struct SPCFwInterParam
    {
        static constexpr auto                   _oxygenCharge   = -0.82;
        static constexpr auto                   _hydrogenCharge = 0.41;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_SPC_FW_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
    };

    struct qSPCFwInterParam
    {
        static constexpr auto                   _oxygenCharge   = -0.84;
        static constexpr auto                   _hydrogenCharge = 0.42;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_QSPC_FW_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
    };

    struct TIP3PInterParam
    {
        static constexpr auto                   _oxygenCharge   = -0.834;
        static constexpr auto                   _hydrogenCharge = 0.417;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_TIP3P_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
    };

    struct OPC3InterParam
    {
        static constexpr auto                   _oxygenCharge   = -0.89517;
        static constexpr auto                   _hydrogenCharge = 0.447585;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_OPC3_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
    };

}   // namespace waterModel

#endif   //  _INTER_WATER_PARAMETERS_TPP_HPP_