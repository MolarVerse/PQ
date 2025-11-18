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

#ifndef _SPC_INTRA_WATER_TPP_HPP_

#define _SPC_INTRA_WATER_TPP_HPP_

#include <cmath>   // for sin

#include "SPCIntraWater.hpp"
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vector3D, norm, operator*, Vec3D

template <class Derived>
void waterModels::SPCIntraWater<Derived>::calculate(
    pq::SimBox&       box,
    pq::PhysicalData& physicalData
)
{
    const auto eqOHDistance = Derived::_eqOHDistance;
    const auto eqHOHAngle   = Derived::_eqHOHAngle;
    const auto kOHBond      = Derived::_forceConstantOHBond;
    const auto kHOHAngle    = Derived::_forceConstantHOHAngle;

    for (auto& water : box.getWaterTypeMolecules())
    {
        auto& oxygen    = water.getAtom(0);
        auto& hydrogen1 = water.getAtom(1);
        auto& hydrogen2 = water.getAtom(2);

        const auto posOxygen    = oxygen.getPosition();
        const auto posHydrogen1 = hydrogen1.getPosition();
        const auto posHydrogen2 = hydrogen2.getPosition();

        auto dOH1 = posOxygen - posHydrogen1;

        box.applyPBC(dOH1);

        const auto dist1          = norm(dOH1);
        const auto deltaDistance1 = dist1 - eqOHDistance;

        auto forceMagnitude1 = -kOHBond * deltaDistance1;

        physicalData.addBondEnergy(-forceMagnitude1 * deltaDistance1 / 2.0);

        forceMagnitude1 /= dist1;

        const auto force1 = forceMagnitude1 * dOH1;
        oxygen.addForce(force1);
        hydrogen1.addForce(-force1);

        auto dOH2 = posOxygen - posHydrogen2;

        box.applyPBC(dOH2);

        const auto dist2          = norm(dOH2);
        const auto deltaDistance2 = dist2 - eqOHDistance;

        auto forceMagnitude2 = -kOHBond * deltaDistance2;

        physicalData.addBondEnergy(-forceMagnitude2 * deltaDistance2 / 2.0);

        forceMagnitude2 /= dist2;

        const auto force2 = forceMagnitude2 * dOH2;
        oxygen.addForce(force2);
        hydrogen2.addForce(-force2);

        auto smF = 0.0;
        if (water->getHybridZone() == SMOOTHING)
            smF = water->getSmoothingFactor();

        physicalData.addVirial(tensorProduct(dist1, force1) * (1 - smF));
        physicalData.addVirial(tensorProduct(dist2, force2) * (1 - smF));

        const auto alpha      = angle(dOH1, dOH2);
        const auto deltaAngle = alpha - eqHOHAngle;

        forceMagnitude3 = -kHOHAngle * deltaAngle;

        physicalData.addAngleEnergy(-forceMagnitude3 * deltaAngle / 2.0);

        const auto normalDistance = dist1 * dist2 * ::sin(alpha);

        auto normalPosition  = cross(dOH2, dOH1);
        normalPosition      /= normalDistance;

        auto force3   = forceMagnitude3 / dist1;
        auto forcexyz = force3 * cross(dOH1, normalPosition);

        oxygen.addForce(-forcexyz);
        hydrogen1.addForce(forcexyz);

        force4   = forceMagnitude3 / dist2;
        forcexyz = force4 * cross(normalPosition, dOH2);

        oxygen.addForce(-forcexyz);
        hydrogen2.addForce(forcexyz);
    }
}

#endif   //  _SPC_INTRA_WATER_TPP_HPP_