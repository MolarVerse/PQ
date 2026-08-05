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

#ifndef _INTER_WATER_PARAMETERS_HPP_

#define _INTER_WATER_PARAMETERS_HPP_

#include <concepts>
#include <memory>
#include <type_traits>

#include "defaults.hpp"
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
        auto state = InterWaterState();

        state._oxygenCharge         = T::_oxygenCharge;
        state._hydrogenCharge       = T::_hydrogenCharge;
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

    struct SPCInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.82;
        static constexpr auto    _hydrogenCharge       = 0.41;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::COULOMB_CUT_OFF_DEFAULT,
                constants::_SPC_LJ_C6_OO_,
                constants::_SPC_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct SPCEInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.8476;
        static constexpr auto    _hydrogenCharge       = 0.4238;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::COULOMB_CUT_OFF_DEFAULT,
                constants::_SPC_E_LJ_C6_OO_,
                constants::_SPC_E_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct SPCFwInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.82;
        static constexpr auto    _hydrogenCharge       = 0.41;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::COULOMB_CUT_OFF_DEFAULT,
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
                defaults::COULOMB_CUT_OFF_DEFAULT,
                constants::_QSPC_FW_LJ_C6_OO_,
                constants::_QSPC_FW_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct SPCDCInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.87362;
        static constexpr auto    _hydrogenCharge       = 0.43681;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::COULOMB_CUT_OFF_DEFAULT,
                constants::_H2O_DC_LJ_C6_OO_,
                constants::_H2O_DC_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct H2ODCInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.9099;
        static constexpr auto    _hydrogenCharge       = 0.45495;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::COULOMB_CUT_OFF_DEFAULT,
                constants::_H2O_DC_LJ_C6_OO_,
                constants::_H2O_DC_LJ_C12_OO_
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
                defaults::COULOMB_CUT_OFF_DEFAULT,
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
                defaults::COULOMB_CUT_OFF_DEFAULT,
                constants::_OPC3_LJ_C6_OO_,
                constants::_OPC3_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct SPCmTRInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.82;
        static constexpr auto    _hydrogenCharge       = 0.41;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::COULOMB_CUT_OFF_DEFAULT,
                constants::_SPC_MTR_LJ_C6_OO_,
                constants::_SPC_MTR_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

    struct TIP3PmTRInterParam
    {
        static constexpr auto    _oxygenCharge         = -0.834;
        static constexpr auto    _hydrogenCharge       = 0.417;
        static constexpr bool    _oxygenOnlyNonCoulomb = true;
        inline static const auto _nonCoulombPairOO =
            potential::LennardJonesPair(
                defaults::COULOMB_CUT_OFF_DEFAULT,
                constants::_TIP3P_MTR_LJ_C6_OO_,
                constants::_TIP3P_MTR_LJ_C12_OO_
            );
        inline static const auto _nonCoulombPairOH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
        inline static const auto _nonCoulombPairHH =
            potential::LennardJonesPair(0.01, 0.0, 0.0);
    };

}   // namespace waterModel

#endif   //  _INTER_WATER_PARAMETERS_HPP_