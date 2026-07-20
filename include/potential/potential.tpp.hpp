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

#ifndef _POTENTIAL_TPP_

#define _POTENTIAL_TPP_

#include <cmath>     // for sqrt
#include <cstdlib>   // for abort

#include "box.hpp"                   // for Box
#include "coulombPotential.hpp"      // for CoulombPotential
#include "hybridSettings.hpp"        // for HybridSettings
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "potential.hpp"

namespace potential
{
    /**
     * @brief Calculate non-bonded inter-molecular interactions between two
     * atoms
     *
     * @details This template function computes both Coulombic and non-Coulombic
     * interactions between two atoms from different molecules. The charge types
     * are determined at compile time using template specialization, allowing
     * for efficient QM/MM hybrid calculations. The function applies periodic
     * boundary conditions, calculates distances, and evaluates both
     * electrostatic and van der Waals interactions within their respective
     * cutoff radii.
     *
     * @tparam ChargeTag1 Charge type for atom1 (QMChargeTag or MMChargeTag)
     * @tparam ChargeTag2 Charge type for atom2 (QMChargeTag or MMChargeTag)
     * @param box Simulation box for periodic boundary conditions
     * @param mol1 First molecule containing atom1
     * @param mol2 Second molecule containing atom2
     * @param atom1 First atom in the interaction pair
     * @param atom2 Second atom in the interaction pair
     * @return std::pair<double, double> Coulomb energy and non-Coulomb energy
     */
    template <typename ChargeTag1, typename ChargeTag2>
    std::pair<double, double> Potential::calculateSingleInteraction(
        const pq::Box &box,
        pq::Molecule  &mol1,
        pq::Molecule  &mol2,
        pq::Atom      &atom1,
        pq::Atom      &atom2
    ) const
    {
        auto coulombEnergy    = 0.0;
        auto nonCoulombEnergy = 0.0;

        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -box.calcShiftVector(dxyz);

        dxyz += txyz;

        const double distanceSquared = normSquared(dxyz);

        if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff();
            distanceSquared < RcCutOff * RcCutOff)
        {
            const double distance   = ::sqrt(distanceSquared);
            const auto   atomType_i = atom1.getAtomType();
            const auto   atomType_j = atom2.getAtomType();

            const auto globalVdwType_i = atom1.getInternalGlobalVDWType();
            const auto globalVdwType_j = atom2.getInternalGlobalVDWType();

            const auto moltype_i = mol1.getMoltype();
            const auto moltype_j = mol2.getMoltype();

            const auto combinedIdx = {
                moltype_i,
                moltype_j,
                atomType_i,
                atomType_j,
                globalVdwType_i,
                globalVdwType_j
            };

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto coulombPreFactor = charge_i * charge_j;

            auto [e, f] =
                _coulombPotential->calculate(distance, coulombPreFactor);
            coulombEnergy = e;

            const auto nonCoulPair =
                _nonCoulombPot->getNonCoulPair(combinedIdx);
            const auto rncCutOff = nonCoulPair->getRadialCutOff();

            if (distance < rncCutOff)
            {
                const auto &[nonCoulE, nonCoulF] =
                    nonCoulPair->calculate(distance);
                nonCoulombEnergy = nonCoulE;

                f += nonCoulF;
            }

            f /= distance;

            const auto forcexyz = f * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);
            atom2.addForce(-forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }

        return {coulombEnergy, nonCoulombEnergy};
    }

    /**
     * @brief Calculate single Coulomb interaction between two atoms
     *
     * @details This template function computes only the electrostatic
     * interaction between two atoms, ignoring van der Waals forces. It's
     * optimized for cases where only Coulombic interactions are needed. The
     * function applies periodic boundary conditions and respects the Coulomb
     * cutoff radius.
     *
     * @tparam ChargeTag1 Charge type for atom1 (QMChargeTag or MMChargeTag)
     * @tparam ChargeTag2 Charge type for atom2 (QMChargeTag or MMChargeTag)
     * @param box Simulation box for periodic boundary conditions
     * @param atom1 First atom in the interaction pair
     * @param atom2 Second atom in the interaction pair
     * @return double Coulomb energy of the interaction
     */
    template <typename ChargeTag1, typename ChargeTag2>
    double Potential::calculateSingleCoulombInteraction(
        const pq::Box &box,
        pq::Atom      &atom1,
        pq::Atom      &atom2
    ) const
    {
        auto coulombEnergy = 0.0;

        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -box.calcShiftVector(dxyz);

        dxyz += txyz;

        const double distanceSquared = normSquared(dxyz);

        if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff();
            distanceSquared < RcCutOff * RcCutOff)
        {
            const double distance = ::sqrt(distanceSquared);

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto coulombPreFactor = charge_i * charge_j;

            auto [e, f] =
                _coulombPotential->calculate(distance, coulombPreFactor);
            coulombEnergy = e;

            f /= distance;

            const auto forcexyz = f * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);
            atom2.addForce(-forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }

        return coulombEnergy;
    }

    /**
     * @brief Calculate one-way non-bonded interactions between two atoms
     *
     * @details This template function is similar to calculateSingleInteraction
     * but applies forces only to the first atom (atom1). This is useful in
     * specific algorithmic contexts where force symmetry is handled elsewhere
     * or when calculating interactions in a one-directional manner. The
     * function still computes both Coulombic and non-Coulombic interactions but
     * only updates forces and shift forces for atom1.
     *
     * @tparam ChargeTag1 Charge type for atom1 (QMChargeTag or MMChargeTag)
     * @tparam ChargeTag2 Charge type for atom2 (QMChargeTag or MMChargeTag)
     * @param box Simulation box for periodic boundary conditions
     * @param mol1 First molecule containing atom1
     * @param mol2 Second molecule containing atom2
     * @param atom1 First atom in the interaction pair (receives forces)
     * @param atom2 Second atom in the interaction pair (no forces applied)
     * @return std::pair<double, double> Coulomb energy and non-Coulomb energy
     */
    template <typename ChargeTag1, typename ChargeTag2>
    std::pair<double, double> Potential::calculateSingleInteractionOneWay(
        const pq::Box &box,
        pq::Molecule  &mol1,
        pq::Molecule  &mol2,
        pq::Atom      &atom1,
        pq::Atom      &atom2
    ) const
    {
        auto coulombEnergy    = 0.0;
        auto nonCoulombEnergy = 0.0;

        const auto xyz_i = atom1.getPosition();
        const auto xyz_j = atom2.getPosition();

        auto dxyz = xyz_i - xyz_j;

        const auto txyz = -box.calcShiftVector(dxyz);

        dxyz += txyz;

        const double distanceSquared = normSquared(dxyz);

        if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff();
            distanceSquared < RcCutOff * RcCutOff)
        {
            const double distance   = ::sqrt(distanceSquared);
            const auto   atomType_i = atom1.getAtomType();
            const auto   atomType_j = atom2.getAtomType();

            const auto globalVdwType_i = atom1.getInternalGlobalVDWType();
            const auto globalVdwType_j = atom2.getInternalGlobalVDWType();

            const auto moltype_i = mol1.getMoltype();
            const auto moltype_j = mol2.getMoltype();

            const auto combinedIdx = {
                moltype_i,
                moltype_j,
                atomType_i,
                atomType_j,
                globalVdwType_i,
                globalVdwType_j
            };

            const auto charge_i = getPartialCharge<ChargeTag1>(atom1);
            const auto charge_j = getPartialCharge<ChargeTag2>(atom2);

            const auto coulombPreFactor = charge_i * charge_j;

            auto [e, f] =
                _coulombPotential->calculate(distance, coulombPreFactor);
            coulombEnergy = e;

            const auto nonCoulPair =
                _nonCoulombPot->getNonCoulPair(combinedIdx);
            const auto rncCutOff = nonCoulPair->getRadialCutOff();

            if (distance < rncCutOff)
            {
                const auto &[nonCoulE, nonCoulF] =
                    nonCoulPair->calculate(distance);
                nonCoulombEnergy = nonCoulE;

                f += nonCoulF;
            }

            f /= distance;

            const auto forcexyz = f * dxyz;

            const auto shiftForcexyz = forcexyz * txyz;

            atom1.addForce(forcexyz);

            atom1.addShiftForce(shiftForcexyz);
        }

        return {coulombEnergy, nonCoulombEnergy};
    }

    /**
     * @brief make shared pointer of the Coulomb potential
     *
     * @tparam T
     * @param p
     */
    template <typename T>
    void Potential::makeCoulombPotential(T p)
    {
        _coulombPotential = std::make_shared<T>(p);
    }

    /**
     * @brief make shared pointer of the non-Coulomb potential
     *
     * @tparam T
     * @param nonCoulombPotential
     */
    template <typename T>
    void Potential::makeNonCoulombPotential(T nonCoulombPotential)
    {
        _nonCoulombPot = std::make_shared<T>(nonCoulombPotential);
    }

    /**
     * @brief Generic template function for getting partial charge from an atom
     *
     * @details This base template should never be called directly. It serves as
     * a fallback that will abort the program if called with an unsupported
     * charge tag type. Only the specialized versions (QMChargeTag and
     * MMChargeTag) should be used.
     *
     * @tparam T Charge tag type (should be QMChargeTag or MMChargeTag)
     * @param atom Reference to the atom from which to extract the charge
     * @return double The partial charge (this implementation aborts)
     *
     * @throws std::abort() Always aborts as this should never be called
     */
    template <typename T>
    double Potential::getPartialCharge(pq::Atom &atom) const
    {
        std::abort();
    }

    /**
     * @brief Template specialization for getting QM partial charge from an atom
     *
     * @details This specialization handles quantum mechanical charge
     * extraction. It first attempts to use the QM charge if available
     * (calculated from QM methods), and falls back to the standard partial
     * charge if no QM charge is present.
     *
     * @param atom Reference to the atom from which to extract the QM charge
     * @return double The QM charge if available, otherwise the standard partial
     * charge
     */
    template <>
    inline double Potential::getPartialCharge<QMChargeTag>(pq::Atom &atom) const
    {
        const auto useQMCharges = settings::HybridSettings::getUseQMCharges();

        if (atom.getQMCharge() && useQMCharges)
            return atom.getQMCharge().value();
        else
            return atom.getPartialCharge();
    }

    /**
     * @brief Template specialization for getting MM partial charge from an atom
     *
     * @details This specialization always returns the standard partial charge
     * used in classical force fields.
     *
     * @param atom Reference to the atom from which to extract the MM charge
     * @return double The molecular mechanics partial charge
     */
    template <>
    inline double Potential::getPartialCharge<MMChargeTag>(pq::Atom &atom) const
    {
        return atom.getPartialCharge();
    }

}   // namespace potential

#endif   // _POTENTIAL_TPP_