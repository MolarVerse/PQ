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

#ifndef FORCE_FIELD__FORCEFIELD_TPP
#define FORCE_FIELD__FORCEFIELD_TPP

#include "forceField.hpp"
#include "potentialSettings.hpp"

namespace ff
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
     * @param coulombPotential
     * @param nonCoulombPotential
     * @param physicalData
     * @param molecule_i
     * @param molecule_j
     * @param atomIndex1
     * @param atomIndex2
     * @param distance
     * @return double
     */
    template <typename T>
    double correctLinker(
        const pot::CoulombPotential &coulombPotential,
        pot::NonCoulombPotential    &nonCoulombPotential,
        physicalData::PhysicalData  &physicalData,
        const molsys::Molecule      *molecule_i,
        const molsys::Molecule      *molecule_j,
        AtomIndex                    atomIndex1,
        AtomIndex                    atomIndex2,
        double                       distance
    )
    {
        const auto charge1 = molecule_i->getPartialCharge(atomIndex1);
        const auto charge2 = molecule_j->getPartialCharge(atomIndex2);

        const auto chargeProduct = charge1 * charge2;

        auto [coulombEnergy, coulombForce] =
            coulombPotential.calculate(distance, chargeProduct);

        if constexpr (std::is_same_v<T, DihedralForceField>)
        {
            const auto scale = settings::PotentialSettings::getScale14Coulomb();

            // NOLINTBEGIN(readability-magic-numbers)
            coulombEnergy *= (1.0 - scale);
            coulombForce  *= (1.0 - scale);
            // NOLINTEND(readability-magic-numbers)
        }

        auto forceMagnitude = -coulombForce;
        physicalData.addCoulombEnergy(-coulombEnergy);

        const auto molType1  = molecule_i->getMoltype();
        const auto molType2  = molecule_j->getMoltype();
        const auto atomType1 = molecule_i->getAtomType(atomIndex1);
        const auto atomType2 = molecule_j->getAtomType(atomIndex2);
        const auto vdwType1  = molecule_i->getInternalGlobalVDWType(atomIndex1);
        const auto vdwType2  = molecule_j->getInternalGlobalVDWType(atomIndex2);

        const std::tuple indices{molType1, molType2, atomType1, atomType2};

        const auto nonCoulombPair =
            nonCoulombPotential.getNonCoulPair(indices, {vdwType1, vdwType2});

        if (distance < nonCoulombPair->get_radialCutoff())
        {
            auto [nonCoulombEnergy, nonCoulombForce] =
                nonCoulombPair->calculate(distance);

            if constexpr (std::is_same_v<T, DihedralForceField>)
            {
                const auto scale = settings::PotentialSettings::getScale14VDW();

                // NOLINTBEGIN(readability-magic-numbers)
                nonCoulombEnergy *= (1.0 - scale);
                nonCoulombForce  *= (1.0 - scale);
                // NOLINTEND(readability-magic-numbers)
            }

            forceMagnitude -= nonCoulombForce;
            physicalData.addNonCoulombEnergy(-nonCoulombEnergy);
        }

        return forceMagnitude;
    }
}   // namespace ff

#endif   // FORCE_FIELD__FORCEFIELD_TPP
