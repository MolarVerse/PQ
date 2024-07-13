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

#ifndef _FORCE_FIELD_TPP_

#define _FORCE_FIELD_TPP_

#include "forceField.hpp"
#include "typeAliases.hpp"

namespace forceField
{
    class DihedralForceField;   // forward declaration

    /**
     * @brief correct coulomb and non-coulomb energy and forces for linker
     * connectivity elements
     *
     * @details with template parameter T, scaling of non-coulomb energy and
     * forces can be applied to dihedrals
     *
     * @tparam T
     * @param coulPot
     * @param nonCoulPot
     * @param physicalData
     * @param molecule1
     * @param molecule2
     * @param atomIndex1
     * @param atomIndex2
     * @param distance
     * @param isDihedral
     * @return double
     */
    template <typename T>
    double correctLinker(
        const pq::CoulombPot &coulPot,
        pq::NonCoulombPot    &nonCoulPot,
        pq::PhysicalData     &physicalData,
        const pq::Molecule   *molecule1,
        const pq::Molecule   *molecule2,
        const size_t          atomIndex1,
        const size_t          atomIndex2,
        const double          distance
    )
    {
        const auto q1 = molecule1->getPartialCharge(atomIndex1);
        const auto q2 = molecule2->getPartialCharge(atomIndex2);

        const auto chargeProduct = q1 * q2;

        auto [coulombEnergy, coulombForce] =
            coulPot.calculate(distance, chargeProduct);

        if constexpr (std::is_same_v<T, DihedralForceField>)
        {
            const auto scale = settings::PotentialSettings::getScale14Coulomb();

            coulombEnergy *= (1.0 - scale);
            coulombForce  *= (1.0 - scale);
        }

        auto forceMagnitude = -coulombForce;
        physicalData.addCoulombEnergy(-coulombEnergy);

        const auto molType1  = molecule1->getMoltype();
        const auto molType2  = molecule2->getMoltype();
        const auto atomType1 = molecule1->getAtomType(atomIndex1);
        const auto atomType2 = molecule2->getAtomType(atomIndex2);
        const auto vdwType1  = molecule1->getInternalGlobalVDWType(atomIndex1);
        const auto vdwType2  = molecule2->getInternalGlobalVDWType(atomIndex2);

        const auto indices =
            {molType1, molType2, atomType1, atomType2, vdwType1, vdwType2};

        const auto nonCoulombPair = nonCoulPot.getNonCoulPair(indices);

        if (distance < nonCoulombPair->getRadialCutOff())
        {
            auto [nonCoulombEnergy, nonCoulombForce] =
                nonCoulombPair->calculate(distance);

            if constexpr (std::is_same_v<T, DihedralForceField>)
            {
                const auto scale = settings::PotentialSettings::getScale14VDW();

                nonCoulombEnergy *= (1.0 - scale);
                nonCoulombForce  *= (1.0 - scale);
            }

            forceMagnitude -= nonCoulombForce;
            physicalData.addNonCoulombEnergy(-nonCoulombEnergy);
        }

        return forceMagnitude;
    }
}   // namespace forceField

#endif   // _FORCE_FIELD_TPP_