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

#ifndef _INTER_WATER_COULOMB_HPP_

#define _INTER_WATER_COULOMB_HPP_

#include <array>

#include "atom.hpp"               // for Atom
#include "guffCoefficients.hpp"   // for guffPairCoefficients
#include "guffPair.hpp"           // for GuffPair
#include "physicalData.hpp"       // for PhysicalData
#include "simulationBox.hpp"      // for SimulationBox
#include "typeAliases.hpp"

namespace waterModel
{
    template <class Derived>
    class InterWaterCoulomb
    {
       public:
        void initGuffPairs();

        void calculate(
            pq::SimBox &,
            pq::PhysicalData &,
            pq::SharedCoulombPot &
        );

       private:
        auto _nonCoulombCutOff = 12.0;
        auto _guffPairOO       = potential::GuffPair();
        auto _guffPairOH       = potential::GuffPair();
        auto _guffPairHH       = potential::GuffPair();

        double calculateSingleInteraction(
            pq::Atom             &atom1,
            pq::Atom             &atom2,
            double                chargeProduct,
            pq::SharedCoulombPot &coulombPotential,
            double                rCutSquared,
            pq::SimBox           &simBox
        );
    };

    class SPCFwInterParam : public InterWaterCoulomb<SPCFwInterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.82;
        static constexpr auto _hydrogenCharge = 0.41;
        static constexpr auto _guffCoefficientsOO =
            constants::_SPC_FW_GUFF_COEFFICIENTS_OO_;
        static constexpr std::array<double, 22> _guffCoefficientsOH{};
        static constexpr std::array<double, 22> _guffCoefficientsHH{};

        friend class InterWaterCoulomb<SPCFwInterParam>;
    };

    class qSPCFwInterParam : public InterWaterCoulomb<qSPCFwInterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.84;
        static constexpr auto _hydrogenCharge = 0.42;

        friend class InterWaterCoulomb<qSPCFwInterParam>;
    };

    class TIP3PInterParam : public InterWaterCoulomb<TIP3PInterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.834;
        static constexpr auto _hydrogenCharge = 0.417;

        friend class InterWaterCoulomb<TIP3PInterParam>;
    };

    class OPC3InterParam : public InterWaterCoulomb<OPC3InterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.89517;
        static constexpr auto _hydrogenCharge = 0.447585;

        friend class InterWaterCoulomb<OPC3InterParam>;
    };

}   // namespace waterModel

#include "interWaterCoulomb.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _INTER_WATER_COULOMB_HPP_