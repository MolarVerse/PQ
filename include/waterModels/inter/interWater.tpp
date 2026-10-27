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

#ifndef _INTER_WATER_TPP_

#define _INTER_WATER_TPP_

#include "atom.hpp"
#include "coulombPotential.hpp"
#include "hybridSettings.hpp"
#include "interWater.hpp"
#include "potential.hpp"
#include "simulationBox.hpp"

namespace waterModel
{
    /**
     * @brief Compute Coulomb and non-Coulomb contributions for one atom pair.
     *
     * @tparam ChargeTag1 Charge tag for atom1.
     * @tparam ChargeTag2 Charge tag for atom2.
     * @tparam Weighted true if the pair is scaled by weight.
     */
    template <typename ChargeTag1, typename ChargeTag2, bool Weighted>
    void InterWaterStrategy::_calculateSingleInteraction(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        const pot::NonCoulombPair                    &nonCoulPair,
        double                                       &coulombEnergy,
        double                                       &nonCoulombEnergy,
        const double                                  weight
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -simulationBox.calcShiftVector(dxyz);

        dxyz += txyz;

        const auto distanceSquared = normSquared(dxyz);

        if (distanceSquared < rCutSquared)
        {
            const auto distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto chargeProduct = charge_i * charge_j;

            auto [e, f] = coulombPotential->calculate(distance, chargeProduct);
            coulombEnergy += pot::applyWeight<Weighted>(e, weight);

            if (distance < nonCoulPair.get_radialCutoff())
            {
                const auto [nonCoulE, nonCoulF] =
                    nonCoulPair.calculate(distance);
                nonCoulombEnergy +=
                    pot::applyWeight<Weighted>(nonCoulE, weight);
                f += nonCoulF;
            }

            f = pot::applyWeight<Weighted>(f, weight);

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
     * @tparam Weighted true if the pair is scaled by weight.
     */
    template <typename ChargeTag1, typename ChargeTag2, bool Weighted>
    void InterWaterStrategy::_calculateSingleCoulombInteraction(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        double                                       &coulombEnergy,
        const double                                  weight
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -simulationBox.calcShiftVector(dxyz);

        dxyz += txyz;

        const auto distanceSquared = normSquared(dxyz);

        if (distanceSquared < rCutSquared)
        {
            const double distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto chargeProduct = charge_i * charge_j;

            auto [e, f] = coulombPotential->calculate(distance, chargeProduct);
            coulombEnergy += pot::applyWeight<Weighted>(e, weight);

            f = pot::applyWeight<Weighted>(f, weight);

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
     * @tparam Weighted true if the pair is scaled by weight.
     */
    template <typename ChargeTag1, typename ChargeTag2, bool Weighted>
    void InterWaterStrategy::_calculateSingleInteractionOneWay(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        const pot::NonCoulombPair                    &nonCoulPair,
        double                                       &coulombEnergy,
        double                                       &nonCoulombEnergy,
        const double                                  weight
    )
    {
        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -simulationBox.calcShiftVector(dxyz);

        dxyz += txyz;

        const auto distanceSquared = normSquared(dxyz);

        if (distanceSquared < rCutSquared)
        {
            const auto distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto chargeProduct = charge_i * charge_j;

            auto [e, f] = coulombPotential->calculate(distance, chargeProduct);
            coulombEnergy += pot::applyWeight<Weighted>(e, weight);

            if (distance < nonCoulPair.get_radialCutoff())
            {
                auto [nonCoulE, nonCoulF] = nonCoulPair.calculate(distance);
                nonCoulombEnergy +=
                    pot::applyWeight<Weighted>(nonCoulE, weight);
                f += nonCoulF;
            }

            f = pot::applyWeight<Weighted>(f, weight);

            f                   /= distance;
            const auto forcexyz  = f * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }
    }

    /**
     * @brief Return the partial charge for a tagged atom.
     *
     * @tparam T Charge tag type.
     */
    template <typename T>
    double InterWaterStrategy::getPartialCharge(molsys::Atom & /*atom*/) const
    {
        std::abort();
    }

    template <>
    inline double InterWaterStrategy::getPartialCharge<pot::QMChargeTag>(
        molsys::Atom &atom
    ) const
    {
        const auto useQMCharges = settings::HybridSettings::getUseQMCharges();

        if (atom.getQMCharge() && useQMCharges)
            return atom.getQMCharge().value();

        return atom.getPartialCharge();
    }

    template <>
    inline double InterWaterStrategy::getPartialCharge<pot::MMChargeTag>(
        molsys::Atom &atom
    ) const
    {
        return atom.getPartialCharge();
    }

    /**
     * @brief Unweighted calculateSingleInteraction (weight 1)
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleInteraction(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        const pot::NonCoulombPair                    &nonCoulPair,
        double                                       &coulombEnergy,
        double                                       &nonCoulombEnergy
    )
    {
        _calculateSingleInteraction<ChargeTag1, ChargeTag2, false>(
            atom1,
            atom2,
            coulombPotential,
            rCutSquared,
            simulationBox,
            nonCoulPair,
            coulombEnergy,
            nonCoulombEnergy,
            1.0
        );
    }

    /**
     * @brief Unweighted calculateSingleCoulombInteraction (weight 1)
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleCoulombInteraction(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        double                                       &coulombEnergy
    )
    {
        _calculateSingleCoulombInteraction<ChargeTag1, ChargeTag2, false>(
            atom1,
            atom2,
            coulombPotential,
            rCutSquared,
            simulationBox,
            coulombEnergy,
            1.0
        );
    }

    /**
     * @brief Unweighted calculateSingleInteractionOneWay (weight 1)
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleInteractionOneWay(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        const pot::NonCoulombPair                    &nonCoulPair,
        double                                       &coulombEnergy,
        double                                       &nonCoulombEnergy
    )
    {
        _calculateSingleInteractionOneWay<ChargeTag1, ChargeTag2, false>(
            atom1,
            atom2,
            coulombPotential,
            rCutSquared,
            simulationBox,
            nonCoulPair,
            coulombEnergy,
            nonCoulombEnergy,
            1.0
        );
    }

    /**
     * @brief calculateSingleInteraction with pair weight
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleInteraction(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        const pot::NonCoulombPair                    &nonCoulPair,
        double                                       &coulombEnergy,
        double                                       &nonCoulombEnergy,
        const double                                  weight
    )
    {
        _calculateSingleInteraction<ChargeTag1, ChargeTag2, true>(
            atom1,
            atom2,
            coulombPotential,
            rCutSquared,
            simulationBox,
            nonCoulPair,
            coulombEnergy,
            nonCoulombEnergy,
            weight
        );
    }

    /**
     * @brief calculateSingleCoulombInteraction with pair weight
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleCoulombInteraction(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        double                                       &coulombEnergy,
        const double                                  weight
    )
    {
        _calculateSingleCoulombInteraction<ChargeTag1, ChargeTag2, true>(
            atom1,
            atom2,
            coulombPotential,
            rCutSquared,
            simulationBox,
            coulombEnergy,
            weight
        );
    }

    /**
     * @brief calculateSingleInteractionOneWay with pair weight
     */
    template <typename ChargeTag1, typename ChargeTag2>
    void InterWaterStrategy::calculateSingleInteractionOneWay(
        molsys::Atom                                 &atom1,
        molsys::Atom                                 &atom2,
        const std::shared_ptr<pot::CoulombPotential> &coulombPotential,
        double                                        rCutSquared,
        const molsys::SimulationBox                  &simulationBox,
        const pot::NonCoulombPair                    &nonCoulPair,
        double                                       &coulombEnergy,
        double                                       &nonCoulombEnergy,
        const double                                  weight
    )
    {
        _calculateSingleInteractionOneWay<ChargeTag1, ChargeTag2, true>(
            atom1,
            atom2,
            coulombPotential,
            rCutSquared,
            simulationBox,
            nonCoulPair,
            coulombEnergy,
            nonCoulombEnergy,
            weight
        );
    }

}   // namespace waterModel

#endif   //  _INTER_WATER_TPP_
