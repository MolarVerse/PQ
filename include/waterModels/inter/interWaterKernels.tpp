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

#ifndef WATER_MODELS__INTER__INTER_WATER_KERNELS_TPP
#define WATER_MODELS__INTER__INTER_WATER_KERNELS_TPP

/**
 * @file interWaterKernels.tpp
 *
 * @brief Definitions of the per-pair kernels of waterModel::InterWaterStrategy.
 *
 * @details These kernels run once per atom pair, so the translation units
 * that contain the pair loops (interWaterBruteForce.cpp,
 * interWaterCellList.cpp) must see the definitions to be able to inline them.
 * They take the molsys::Box directly (hoisted out of the loops by the callers)
 * so the periodic shift is a single virtual call without an additional
 * out-of-line SimulationBox wrapper. The definitions need the complete type
 * of molsys::Box, which is why they are NOT part of interWater.hpp or
 * interWater.tpp: including this file from a public header would pull
 * box.hpp into all of its dependents again.
 *
 * Only include this file from .cpp files (the pair loops and unit tests).
 */

#include <cmath>
#include <memory>

#include "atom.hpp"
#include "box.hpp"
#include "coulombPotential.hpp"
#include "interWater.hpp"
#include "nonCoulombPair.hpp"

namespace waterModel
{
    /**
     * @brief Compute Coulomb and non-Coulomb contributions for one atom pair.
     *
     * @tparam ChargeTag1 Charge tag for atom1.
     * @tparam ChargeTag2 Charge tag for atom2.
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleInteraction(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::Box                            &box,
        const pot::NonCoulombPair                    &nonCoulPair,
        double                                       &coulombEnergy,
        double                                       &nonCoulombEnergy
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -box.calcShiftVector(dxyz);

        dxyz += txyz;

        const auto distanceSquared = normSquared(dxyz);

        if (distanceSquared < rCutSquared)
        {
            const auto distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto chargeProduct = charge_i * charge_j;

            auto [e, f] = coulombPotential->calculate(distance, chargeProduct);
            coulombEnergy += e;

            if (distance < nonCoulPair.get_radialCutoff())
            {
                const auto [nonCoulE, nonCoulF] =
                    nonCoulPair.calculate(distance);
                nonCoulombEnergy += nonCoulE;
                f                += nonCoulF;
            }

            f                   /= distance;
            const auto forcexyz  = f * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);
            atom2.addForce(-forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }
    }

    /**
     * @brief Compute Coulomb contribution for one atom pair.
     *
     * @tparam ChargeTag1 Charge tag for atom1.
     * @tparam ChargeTag2 Charge tag for atom2.
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleCoulombInteraction(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::Box                            &box,
        double                                       &coulombEnergy
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -box.calcShiftVector(dxyz);

        dxyz += txyz;

        const auto distanceSquared = normSquared(dxyz);

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

    /**
     * @brief Compute one-way Coulomb and non-Coulomb contributions.
     *
     * @tparam ChargeTag1 Charge tag for atom1.
     * @tparam ChargeTag2 Charge tag for atom2.
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleInteractionOneWay(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::Box                            &box,
        const pot::NonCoulombPair                    &nonCoulPair,
        double                                       &coulombEnergy,
        double                                       &nonCoulombEnergy
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -box.calcShiftVector(dxyz);

        dxyz += txyz;

        const auto distanceSquared = normSquared(dxyz);

        if (distanceSquared < rCutSquared)
        {
            const auto distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto chargeProduct = charge_i * charge_j;

            auto [e, f] = coulombPotential->calculate(distance, chargeProduct);
            coulombEnergy += e;

            if (distance < nonCoulPair.get_radialCutoff())
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

}   // namespace waterModel

#endif   //  WATER_MODELS__INTER__INTER_WATER_KERNELS_TPP
