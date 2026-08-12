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

#ifndef _SPC_INTRA_WATER_TPP_

#define _SPC_INTRA_WATER_TPP_

#include <cmath>   // for sin

#include "SPCIntraWater.hpp"    // for SPCIntraWater
#include "hybridSettings.hpp"   // for HybridSettings
#include "physicalData.hpp"     // for PhysicalData
#include "simulationBox.hpp"    // for SimulationBox

/**
 * @brief Calculate intramolecular SPC water interactions for all water
 * molecules.
 *
 * @details For each water molecule, computes:
 * - Two O-H bond energies and forces using a harmonic potential.
 * - One H-O-H angle energy and force using a harmonic potential.
 * - Applies periodic boundary conditions to bond vectors.
 * - Accumulates bond energy, angle energy, and virial contributions.
 * - Handles hybrid-zone smoothing factors if applicable.
 *
 * @param box The simulation box containing water molecules and PBC settings.
 * @param physicalData The physical data container to accumulate energies and
 * virial.
 *
 * @tparam Derived The derived SPC water model class providing equilibrium
 * distances, angles, and force constants.
 *
 * @note Inactive molecules are skipped.
 */
void waterModel::SPCIntraWater::calculate(
    simulationBox::SimulationBox& box,
    physicalData::PhysicalData&   physicalData
)
{
    auto _ = scoped("Calculate Potential");

    const auto eqOHDistance = getEqOHDistance();
    const auto eqHOHAngle   = getEqHOHAngle();
    const auto kOHBond      = getForceConstantOHBond();
    const auto kHOHAngle    = getForceConstantHOHAngle();

    for (auto& water : box.getWaterTypeMolecules())
    {
        auto& oxygen    = water.getAtom(0);
        auto& hydrogen1 = water.getAtom(1);
        auto& hydrogen2 = water.getAtom(2);

        const auto posO  = oxygen.getPosition();
        const auto posH1 = hydrogen1.getPosition();
        const auto posH2 = hydrogen2.getPosition();

        auto dOH1 = posO - posH1;

        box.applyPBC(dOH1);

        const auto distOH1          = norm(dOH1);
        const auto deltaDistanceOH1 = distOH1 - eqOHDistance;

        auto forceMagnitudeOH1 = -kOHBond * deltaDistanceOH1;

        physicalData.addBondEnergy(-forceMagnitudeOH1 * deltaDistanceOH1 / 2.0);

        forceMagnitudeOH1 /= distOH1;

        const auto forceOH1 = forceMagnitudeOH1 * dOH1;
        oxygen.addForce(forceOH1);
        hydrogen1.addForce(-forceOH1);

        auto dOH2 = posO - posH2;

        box.applyPBC(dOH2);

        const auto distOH2          = norm(dOH2);
        const auto deltaDistanceOH2 = distOH2 - eqOHDistance;

        auto forceMagnitudeOH2 = -kOHBond * deltaDistanceOH2;

        physicalData.addBondEnergy(-forceMagnitudeOH2 * deltaDistanceOH2 / 2.0);

        forceMagnitudeOH2 /= distOH2;

        const auto forceOH2 = forceMagnitudeOH2 * dOH2;
        oxygen.addForce(forceOH2);
        hydrogen2.addForce(-forceOH2);

        using enum simulationBox::HybridZone;
        using enum settings::SmoothingMethod;

        auto       smF       = 0.0;
        const auto smoothing = settings::HybridSettings::getSmoothingMethod();

        if (smoothing == HOTSPOT && water.getHybridZone() == SMOOTHING)
            smF = water.getSmoothingFactor();

        physicalData.addVirial(tensorProduct(dOH1, forceOH1) * (1 - smF));
        physicalData.addVirial(tensorProduct(dOH2, forceOH2) * (1 - smF));

        const auto alpha      = angle(dOH1, dOH2);
        const auto deltaAngle = alpha - eqHOHAngle;

        auto forceMagnitudeAngle = -kHOHAngle * deltaAngle;

        physicalData.addAngleEnergy(-forceMagnitudeAngle * deltaAngle / 2.0);

        const auto normalDistance = distOH1 * distOH2 * ::sin(alpha);

        auto normalPosition  = cross(dOH2, dOH1);
        normalPosition      /= normalDistance;

        auto forceAngle = forceMagnitudeAngle / (distOH1 * distOH1);
        auto forcexyz   = forceAngle * cross(dOH1, normalPosition);

        oxygen.addForce(-forcexyz);
        hydrogen1.addForce(forcexyz);

        forceAngle = forceMagnitudeAngle / (distOH2 * distOH2);
        forcexyz   = forceAngle * cross(normalPosition, dOH2);

        oxygen.addForce(-forcexyz);
        hydrogen2.addForce(forcexyz);
    }
}

#endif   //  _SPC_INTRA_WATER_TPP_
