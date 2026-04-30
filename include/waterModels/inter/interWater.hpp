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
    class InterWater
    {
       public:
        virtual void initGuffPairs() {}

        virtual void calculate(
            pq::SimBox &,
            pq::PhysicalData &,
            const pq::SharedCoulombPot &
        )
        {
        }
    };

    template <class Derived>
    class InterWaterImpl : public InterWater
    {
       public:
        virtual void initGuffPairs() override;

        virtual void calculate(
            pq::SimBox &,
            pq::PhysicalData &,
            const pq::SharedCoulombPot &
        ) override;

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

    class SPCFwInterParam : public InterWaterImpl<SPCFwInterParam>
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

        friend class InterWaterImpl<SPCFwInterParam>;
    };

    class qSPCFwInterParam : public InterWaterImpl<qSPCFwInterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.84;
        static constexpr auto _hydrogenCharge = 0.42;

        friend class InterWaterImpl<qSPCFwInterParam>;
    };

    class TIP3PInterParam : public InterWaterImpl<TIP3PInterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.834;
        static constexpr auto _hydrogenCharge = 0.417;

        friend class InterWaterImpl<TIP3PInterParam>;
    };

    class OPC3InterParam : public InterWaterImpl<OPC3InterParam>
    {
       private:
        static constexpr auto _oxygenCharge   = -0.89517;
        static constexpr auto _hydrogenCharge = 0.447585;

        friend class InterWaterImpl<OPC3InterParam>;
    };

}   // namespace waterModel

#include "interWater.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   //  _INTER_WATER_HPP_