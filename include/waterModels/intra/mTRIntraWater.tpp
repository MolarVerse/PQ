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

#ifndef _MTR_INTRA_WATER_TPP_

#define _MTR_INTRA_WATER_TPP_

#include <cmath>

#include "hybridSettings.hpp"   // for HybridSettings
#include "mTRIntraWater.hpp"    // for MTRIntraWater
#include "physicalData.hpp"     // for PhysicalData
#include "simulationBox.hpp"    // for SimulationBox

void waterModel::MTRIntraWater::calculate(
    simulationBox::SimulationBox& box,
    physicalData::PhysicalData&   physicalData
)
{
    startTimingsSection("Calculate Potential");

    const auto eqOHDistance = getEqOHDistance();
    const auto eqHHDistance = getEqHHDistance();
    const auto DOH          = getDOH();
    const auto alpha        = getAlpha();
    const auto beta         = getBeta();
    const auto Ltt          = getLtt();
    const auto Lrt          = getLrt();
    const auto Lrr          = getLrr();

    for (auto& water : box.getWaterTypeMolecules())
    {
        auto& oxygen    = water.getAtom(0);
        auto& hydrogen1 = water.getAtom(1);
        auto& hydrogen2 = water.getAtom(2);

        const auto posO  = oxygen.getPosition();
        const auto posH1 = hydrogen1.getPosition();
        const auto posH2 = hydrogen2.getPosition();

        auto dOH1 = posO - posH1;
        auto dOH2 = posO - posH2;
        auto dHH  = posH1 - posH2;

        box.applyPBC(dOH1);
        box.applyPBC(dOH2);
        box.applyPBC(dHH);

        const auto distOH1 = norm(dOH1);
        const auto distOH2 = norm(dOH2);
        const auto distHH  = norm(dHH);

        const auto deltaOH1 = distOH1 - eqOHDistance;
        const auto deltaOH2 = distOH2 - eqOHDistance;
        const auto deltaHH  = distHH - eqHHDistance;

        const auto expFactorOH1 = std::exp(-alpha * deltaOH1);
        const auto expFactorOH2 = std::exp(-alpha * deltaOH2);

        const auto morseFactorOH1 = 1.0 - expFactorOH1;
        const auto morseFactorOH2 = 1.0 - expFactorOH2;

        const auto gaussianFactor =
            std::exp(-beta * (deltaOH1 * deltaOH1 + deltaOH2 * deltaOH2));

        const auto bondEnergy = DOH * (morseFactorOH1 * morseFactorOH1 +
                                       morseFactorOH2 * morseFactorOH2);

        auto angleEnergy  = Lrr * deltaOH1 * deltaOH2;
        angleEnergy      += Lrt * (deltaOH1 + deltaOH2) * deltaHH;
        angleEnergy      += 0.5 * Ltt * deltaHH * deltaHH;
        angleEnergy      *= gaussianFactor;

        auto fOH1  = 2.0 * DOH * alpha * morseFactorOH1 * expFactorOH1;
        fOH1      += gaussianFactor * (Lrr * deltaOH2 + Lrt * deltaHH) -
                2.0 * beta * deltaOH1 * angleEnergy;
        auto fOH2  = 2.0 * DOH * alpha * morseFactorOH2 * expFactorOH2;
        fOH2      += gaussianFactor * (Lrr * deltaOH1 + Lrt * deltaHH) -
                2.0 * beta * deltaOH2 * angleEnergy;
        const auto fAngle =
            gaussianFactor * (Lrt * (deltaOH1 + deltaOH2) + Ltt * deltaHH);

        const auto forceOH1   = fOH1 * dOH1 / distOH1;
        const auto forceOH2   = fOH2 * dOH2 / distOH2;
        const auto forceAngle = fAngle * dHH / distHH;

        // clang-format off
        oxygen.addForce(   - forceOH1 - forceOH2             );
        hydrogen1.addForce(+ forceOH1            - forceAngle);
        hydrogen2.addForce(           + forceOH2 + forceAngle);
        // clang-format on

        using enum simulationBox::HybridZone;
        using enum settings::SmoothingMethod;

        auto       smF       = 0.0;
        const auto smoothing = settings::HybridSettings::getSmoothingMethod();

        if (smoothing == HOTSPOT && water.getHybridZone() == SMOOTHING)
            smF = water.getSmoothingFactor();

        physicalData.addVirial(tensorProduct(dOH1, forceOH1) * (1 - smF));
        physicalData.addVirial(tensorProduct(dOH2, forceOH2) * (1 - smF));
        physicalData.addVirial(tensorProduct(dHH, forceAngle) * (1 - smF));

        physicalData.addBondEnergy(bondEnergy);
        physicalData.addAngleEnergy(angleEnergy);
    }

    stopTimingsSection("Calculate Potential");
}

#endif   //  _MTR_INTRA_WATER_TPP_