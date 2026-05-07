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
#include <memory>
#include <type_traits>

#include "guffCoefficients.hpp"   // for Guff coefficients
#include "interWater.hpp"         // for InterWater
#include "lennardJonesPair.hpp"   // for LennardJonesPair

namespace waterModel
{
    template <class T>
    concept InterWaterParameterClass = requires {
        { T::_oxygenCharge } -> std::convertible_to<double>;
        { T::_hydrogenCharge } -> std::convertible_to<double>;
        { T::_oxygenOnlyNonCoulomb } -> std::convertible_to<bool>;
        { T::_nonCoulombPairOO };
        { T::_nonCoulombPairOH };
        { T::_nonCoulombPairHH };
    };

    template <InterWaterParameterClass T>
    inline InterWaterState makeInterWaterState()
    {
        auto       state          = InterWaterState();
        const auto oxygenCharge   = T::_oxygenCharge;
        const auto hydrogenCharge = T::_hydrogenCharge;

        state._chargeProductOO      = oxygenCharge * oxygenCharge;
        state._chargeProductOH      = oxygenCharge * hydrogenCharge;
        state._chargeProductHH      = hydrogenCharge * hydrogenCharge;
        state._oxygenOnlyNonCoulomb = T::_oxygenOnlyNonCoulomb;
        state._nonCoulombPairOO =
            std::make_unique<std::decay_t<decltype(T::_nonCoulombPairOO)>>(
                T::_nonCoulombPairOO
            );
        state._nonCoulombPairOH =
            std::make_unique<std::decay_t<decltype(T::_nonCoulombPairOH)>>(
                T::_nonCoulombPairOH
            );
        state._nonCoulombPairHH =
            std::make_unique<std::decay_t<decltype(T::_nonCoulombPairHH)>>(
                T::_nonCoulombPairHH
            );

        return state;
    }

    struct SPCFwInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.82;
        static constexpr auto    _hydrogenCharge       = 0.41;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::_COULOMB_CUT_OFF_DEFAULT_,
                constants::_SPC_FW_LJ_C6_OO_,
                constants::_SPC_FW_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct qSPCFwInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.84;
        static constexpr auto    _hydrogenCharge       = 0.42;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::_COULOMB_CUT_OFF_DEFAULT_,
                constants::_QSPC_FW_LJ_C6_OO_,
                constants::_QSPC_FW_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct TIP3PInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.834;
        static constexpr auto    _hydrogenCharge       = 0.417;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::_COULOMB_CUT_OFF_DEFAULT_,
                constants::_TIP3P_LJ_C6_OO_,
                constants::_TIP3P_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct OPC3InterParam
    {
        static constexpr auto    _oxygenCharge         = -0.89517;
        static constexpr auto    _hydrogenCharge       = 0.447585;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::_COULOMB_CUT_OFF_DEFAULT_,
                constants::_OPC3_LJ_C6_OO_,
                constants::_OPC3_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

}   // namespace waterModel

#endif   //  _INTER_WATER_PARAMETERS_TPP_HPP_