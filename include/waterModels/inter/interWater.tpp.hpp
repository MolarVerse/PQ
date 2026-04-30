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

#include <utility>

#include "coulombPotential.hpp"
#include "guffPair.hpp"            // for GuffPair
#include "interWater.hpp"          // for InterWater
#include "physicalData.hpp"        // for PhysicalData
#include "potentialSettings.hpp"   // for PotentialSettings
#include "simulationBox.hpp"       // for SimulationBox
#include "typeAliases.hpp"
#include "vector3d.hpp"   // for Vector3D, norm, operator*, Vec3D

template <class Derived>
void waterModel::InterWaterImpl<Derived>::initGuffPairs()
{
    _nonCoulombCutOff =
        settings::PotentialSettings::getNonCoulombRadiusCutOff().value_or(
            settings::PotentialSettings::getCoulombRadiusCutOff()
        );

    const auto makeGuffPair = [this](const std::vector<double> &coefficients)
    { return potential::GuffPair{_nonCoulombCutOff, coefficients}; };

    const auto finalizeCutOff = [this](auto &guffPair)
    {
        const auto [energyCutOff, forceCutOff] =
            guffPair.calculate(_nonCoulombCutOff);
        guffPair.setEnergyCutOff(energyCutOff);
        guffPair.setForceCutOff(forceCutOff);
    };

    _GuffPairOO = makeGuffPair(Derived::_guffCoefficientsOO);
    _GuffPairOH = makeGuffPair(Derived::_guffCoefficientsOH);
    _GuffPairHH = makeGuffPair(Derived::_guffCoefficientsHH);

    finalizeCutOff(_GuffPairOO);
    finalizeCutOff(_GuffPairOH);
    finalizeCutOff(_GuffPairHH);
}

template <class Derived>
void waterModel::InterWaterImpl<Derived>::calculate(
    pq::SimBox           &simBox,
    pq::PhysicalData     &physicalData,
    const pq::SharedCoulombPot &coulombPotential
)
{
    const auto oxygenCharge    = Derived::_oxygenCharge;
    const auto hydrogenCharge  = Derived::_hydrogenCharge;
    const auto chargeProductOO = oxygenCharge * oxygenCharge;
    const auto chargeProductOH = oxygenCharge * hydrogenCharge;
    const auto chargeProductHH = hydrogenCharge * hydrogenCharge;

    const auto rCut        = pq::CoulombPot::getCoulombRadiusCutOff();
    const auto rCutSquared = rCut * rCut;

    auto totalCoulombEnergy    = 0.0;
    auto totalNonCoulombEnergy = 0.0;

    size_t i = 0;
    for (auto &water1 : simBox.getWaterTypeMolecules())
    {
        if (!water1.isActive())
            continue;

        size_t j = 0;
        for (auto &water2 : simBox.getWaterTypeMolecules())
        {
            // avoid double counting and self interaction
            if (j >= i)
                break;

            if (!water2.isActive())
            {
                ++j;
                continue;
            }

            auto &oxygen1   = water1.getAtom(0);
            auto &oxygen2   = water2.getAtom(0);
            auto &hydrogen1 = water1.getAtom(1);
            auto &hydrogen2 = water1.getAtom(2);
            auto &hydrogen3 = water2.getAtom(1);
            auto &hydrogen4 = water2.getAtom(2);

            const auto singleInteraction =
                [&](pq::Atom                  &atomA,
                    pq::Atom                  &atomB,
                    const double               chargeProduct,
                    const potential::GuffPair &guffPair)
            {
                const auto [coulE, nonCoulE] = calculateSingleInteraction(
                    atomA,
                    atomB,
                    chargeProduct,
                    coulombPotential,
                    rCutSquared,
                    simBox,
                    guffPair
                );

                totalCoulombEnergy    += coulE;
                totalNonCoulombEnergy += nonCoulE;
            };

            // O-O interaction
            singleInteraction(oxygen1, oxygen2, chargeProductOO, _guffPairOO);

            // O-H interactions
            singleInteraction(oxygen1, hydrogen3, chargeProductOH, _guffPairOH);
            singleInteraction(oxygen1, hydrogen4, chargeProductOH, _guffPairOH);
            singleInteraction(oxygen2, hydrogen1, chargeProductOH, _guffPairOH);
            singleInteraction(oxygen2, hydrogen2, chargeProductOH, _guffPairOH);

            // H-H interactions
            // clang-format off
            singleInteraction(hydrogen1, hydrogen3, chargeProductHH, _guffPairHH);
            singleInteraction(hydrogen1, hydrogen4, chargeProductHH, _guffPairHH);
            singleInteraction(hydrogen2, hydrogen3, chargeProductHH, _guffPairHH);
            singleInteraction(hydrogen2, hydrogen4, chargeProductHH, _guffPairHH);
            // clang-format on

            ++j;
        }
        ++i;
    }

    physicalData.addCoulombEnergy(totalCoulombEnergy);
    physicalData.addNonCoulombEnergy(totalNonCoulombEnergy);
}

template <class Derived>
std::pair<double, double> waterModel::InterWaterImpl<Derived>::
    calculateSingleInteraction(
        pq::Atom                   &atom1,
        pq::Atom                   &atom2,
        const double                chargeProduct,
        const pq::SharedCoulombPot &coulombPotential,
        const double                rCutSquared,
        const pq::SimBox           &simBox,
        const potential::GuffPair  &guffPair
    )
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto xyz_i = atom1.getPosition();
    const auto xyz_j = atom2.getPosition();

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -simBox.calcShiftVector(dxyz);

    dxyz += txyz;

    const double distanceSquared = normSquared(dxyz);

    if (distanceSquared < rCutSquared)
    {
        const double distance = ::sqrt(distanceSquared);

        auto [e, f]   = coulombPotential->calculate(distance, chargeProduct);
        coulombEnergy = e;

        if (distance < _nonCoulombCutOff)
        {
            auto [nonCoulE, nonCoulF]  = guffPair.calculate(distance);
            nonCoulombEnergy           = nonCoulE;
            f                         += nonCoulF;
        }

        f                   /= distance;
        const auto forcexyz  = f * dxyz;

        const auto shiftForcexyz = forcexyz * txyz;

        atom1.addForce(forcexyz);
        atom2.addForce(-forcexyz);

        atom1.addShiftForce(shiftForcexyz);
    }

    return {coulombEnergy, nonCoulombEnergy};
}

#endif   //  _INTER_WATER_TPP_HPP_