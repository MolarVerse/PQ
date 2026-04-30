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

#ifndef _INTER_WATER_HPP_

#define _INTER_WATER_HPP_

#include <utility>
#include <vector>

#include "atom.hpp"                // for Atom
#include "defaults.hpp"            // for defaults
#include "guffCoefficients.hpp"    // for guffPairCoefficients
#include "guffPair.hpp"            // for GuffPair
#include "physicalData.hpp"        // for PhysicalData
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox
#include "typeAliases.hpp"

namespace waterModel
{
    template <class Derived>
    class InterWater
    {
       public:
        void initGuffPairs();

        void calculate(
            pq::SimBox &,
            pq::PhysicalData &,
            pq::SharedCoulombPot &
        );

       private:
        auto _nonCoulombCutOff = defaults::_COULOMB_CUT_OFF_DEFAULT_;
        auto _guffPairOO       = potential::GuffPair();
        auto _guffPairOH       = potential::GuffPair();
        auto _guffPairHH       = potential::GuffPair();

        std::pair<double, double> calculateSingleInteraction(
            pq::Atom                   &atom1,
            pq::Atom                   &atom2,
            const double                chargeProduct,
            const pq::SharedCoulombPot &coulombPotential,
            const double                rCutSquared,
            const pq::SimBox           &simBox,
            const potential::GuffPair  &guffPair
        );
    };

    class SPCFwInterParam : public InterWater<SPCFwInterParam>
    {
       private:
        static constexpr auto                   _oxygenCharge   = -0.82;
        static constexpr auto                   _hydrogenCharge = 0.41;
        inline static const std::vector<double> _guffCoefficientsOO =
            constants::_SPC_FW_GUFF_COEFFICIENTS_OO_;
        inline static const std::vector<double> _guffCoefficientsOH =
            constants::_ZERO_GUFF_COEFFICIENTS_;
        inline static const std::vector<double> _guffCoefficientsHH =
            constants::_ZERO_GUFF_COEFFICIENTS_;

        friend class InterWater<SPCFwInterParam>;
    };

    class qSPCFwInterParam : public InterWater<qSPCFwInterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.84;
        static constexpr auto _hydrogenCharge = 0.42;

        friend class InterWater<qSPCFwInterParam>;
    };

    class TIP3PInterParam : public InterWater<TIP3PInterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.834;
        static constexpr auto _hydrogenCharge = 0.417;

        friend class InterWater<TIP3PInterParam>;
    };

    class OPC3InterParam : public InterWater<OPC3InterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.89517;
        static constexpr auto _hydrogenCharge = 0.447585;

        friend class InterWater<OPC3InterParam>;
    };

}   // namespace waterModel

#include "interWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _INTER_WATER_HPP_