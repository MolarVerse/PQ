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

#ifndef WATER_MODELS__INTER__INTER_WATER_PARAMTERS_HPP
#define WATER_MODELS__INTER__INTER_WATER_PARAMTERS_HPP

#include <concepts>
#include <memory>
#include <type_traits>

#include "defaults.hpp"
#include "guffCoefficients.hpp"
#include "interWater.hpp"
#include "lennardJonesPair.hpp"
#include "units.hpp"

namespace waterModel
{
    template <typename T>
    concept InterWaterParameterClass = requires {
        { T::oxygenCharge } -> std::convertible_to<Charge>;
        { T::hydrogenCharge } -> std::convertible_to<Charge>;
        { T::oxygenOnlyNonCoulomb } -> std::convertible_to<bool>;
        { T::nonCoulombPairOO };
        { T::nonCoulombPairOH };
        { T::nonCoulombPairHH };
    };

    template <InterWaterParameterClass T>
    inline InterWaterState makeInterWaterState()
    {
        auto state = InterWaterState();

        state.oxygenCharge         = T::oxygenCharge;
        state.hydrogenCharge       = T::hydrogenCharge;
        state.oxygenOnlyNonCoulomb = T::oxygenOnlyNonCoulomb;
        state.nonCoulombPairOO =
            std::make_unique<std::decay_t<decltype(T::nonCoulombPairOO)>>(
                T::nonCoulombPairOO
            );
        state.nonCoulombPairOH =
            std::make_unique<std::decay_t<decltype(T::nonCoulombPairOH)>>(
                T::nonCoulombPairOH
            );
        state.nonCoulombPairHH =
            std::make_unique<std::decay_t<decltype(T::nonCoulombPairHH)>>(
                T::nonCoulombPairHH
            );

        return state;
    }

    struct SPCInterParam
    {
        static constexpr auto    oxygenCharge   = -0.82 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.41 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            SPC_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct SPCEInterParam
    {
        static constexpr auto    oxygenCharge   = -0.8476 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.4238 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            SPC_E_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct SPCFwInterParam
    {
        static constexpr auto    oxygenCharge   = -0.82 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.41 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            SPC_FW_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct qSPCFwInterParam
    {
        static constexpr auto    oxygenCharge   = -0.84 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.42 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            QSPC_FW_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct SPCDCInterParam
    {
        static constexpr auto    oxygenCharge   = -0.87362 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.43681 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            SPC_DC_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct H2ODCInterParam
    {
        static constexpr auto    oxygenCharge   = -0.9099 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.45495 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            H2O_DC_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct TIP3PInterParam
    {
        static constexpr auto    oxygenCharge   = -0.834 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.417 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            TIP3P_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct OPC3InterParam
    {
        static constexpr auto    oxygenCharge   = -0.89517 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.447585 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            OPC3_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct SPCmTRInterParam
    {
        static constexpr auto    oxygenCharge   = -0.82 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.41 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            SPC_MTR_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

    struct TIP3PmTRInterParam
    {
        static constexpr auto    oxygenCharge   = -0.834 * elementary_charge;
        static constexpr auto    hydrogenCharge = 0.417 * elementary_charge;
        static constexpr bool    oxygenOnlyNonCoulomb = true;
        inline static const auto nonCoulombPairOO     = pot::LennardJonesPair(
            defaults::COULOMB_CUT_OFF_DEFAULT,
            TIP3P_MTR_LJ_PARAMS_OO
        );
        inline static const auto nonCoulombPairOH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
        inline static const auto nonCoulombPairHH =
            pot::LennardJonesPair(0.01 * angstrom, LJParams{});
    };

}   // namespace waterModel

#endif   //  WATER_MODELS__INTER__INTER_WATER_PARAMTERS_HPP
