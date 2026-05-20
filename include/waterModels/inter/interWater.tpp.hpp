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

#ifndef _INTER_WATER_TPP_HPP_

#define _INTER_WATER_TPP_HPP_

#include "atom.hpp"               // for Atom
#include "coulombPotential.hpp"   // for CoulombPotential
#include "hybridSettings.hpp"     // for HybridSettings
#include "interWater.hpp"
#include "physicalData.hpp"        // for PhysicalData
#include "potential.hpp"           // for ChargeTag
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox
#include "typeAliases.hpp"
#include "vector3d.hpp"   // for normSquared

namespace waterModel
{
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleInteraction(
        pq::Atom                   &atom1,
        pq::Atom                   &atom2,
        const pq::SharedCoulombPot &coulombPotential,
        const double                rCutSquared,
        const pq::SimBox           &simBox,
        const pq::NonCoulPair      &nonCoulPair,
        double                     &coulombEnergy,
        double                     &nonCoulombEnergy
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -simBox.calcShiftVector(dxyz);

        dxyz += txyz;

        const double distanceSquared = normSquared(dxyz);

        if (distanceSquared < rCutSquared)
        {
            const double distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto chargeProduct = charge_i * charge_j;

            auto [e, f] = coulombPotential->calculate(distance, chargeProduct);
            coulombEnergy += e;

            if (distance < nonCoulPair.getRadialCutOff())
            {
                auto [nonCoulE, nonCoulF]  = nonCoulPair.calculate(distance);
                nonCoulombEnergy          += nonCoulE;
                f                         += nonCoulF;
            }

            f                   /= distance;
            const auto forcexyz  = f * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);
            atom2.addForce(-forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }
    }

    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleCoulombInteraction(
        pq::Atom                   &atom1,
        pq::Atom                   &atom2,
        const pq::SharedCoulombPot &coulombPotential,
        const double                rCutSquared,
        const pq::SimBox           &simBox,
        double                     &coulombEnergy
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -simBox.calcShiftVector(dxyz);

        dxyz += txyz;

        const double distanceSquared = normSquared(dxyz);

        if (distanceSquared < rCutSquared)
        {
            const double distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto chargeProduct = charge_i * charge_j;

            auto [e, f] = coulombPotential->calculate(distance, chargeProduct);
            coulombEnergy += e;

            f                   /= distance;
            const auto forcexyz  = f * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);
            atom2.addForce(-forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }
    }

    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleInteractionOneWay(
        pq::Atom                   &atom1,
        pq::Atom                   &atom2,
        const pq::SharedCoulombPot &coulombPotential,
        const double                rCutSquared,
        const pq::SimBox           &simBox,
        const pq::NonCoulPair      &nonCoulPair,
        double                     &coulombEnergy,
        double                     &nonCoulombEnergy
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -simBox.calcShiftVector(dxyz);

        dxyz += txyz;

        const double distanceSquared = normSquared(dxyz);

        if (distanceSquared < rCutSquared)
        {
            const double distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto chargeProduct = charge_i * charge_j;

            auto [e, f] = coulombPotential->calculate(distance, chargeProduct);
            coulombEnergy += e;

            if (distance < nonCoulPair.getRadialCutOff())
            {
                auto [nonCoulE, nonCoulF]  = nonCoulPair.calculate(distance);
                nonCoulombEnergy          += nonCoulE;
                f                         += nonCoulF;
            }

            f                   /= distance;
            const auto forcexyz  = f * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }
    }

    template <typename T>
    double InterWaterStrategy::getPartialCharge(pq::Atom &atom) const
    {
        std::abort();
    }

    template <>
    inline double InterWaterStrategy::getPartialCharge<potential::QMChargeTag>(
        pq::Atom &atom
    ) const
    {
        const auto useQMCharges = settings::HybridSettings::getUseQMCharges();

        if (atom.getQMCharge() && useQMCharges)
            return atom.getQMCharge().value();
        else
            return atom.getPartialCharge();
    }

    template <>
    inline double InterWaterStrategy::getPartialCharge<potential::MMChargeTag>(
        pq::Atom &atom
    ) const
    {
        return atom.getPartialCharge();
    }

}   // namespace waterModel

#endif   //  _INTER_WATER_TPP_HPP_