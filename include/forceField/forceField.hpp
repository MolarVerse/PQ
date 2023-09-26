/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _FORCE_FIELD_HPP_

#define _FORCE_FIELD_HPP_

#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "physicalData.hpp"          // for PhysicalData
#include "potentialSettings.hpp"     // for PotentialSettings

#include <cstddef>   // for size_t
#include <memory>    // for __shared_ptr_access, shared_ptr

namespace forceField
{
    class DihedralForceField;   // forward declaration
}

namespace forceField
{
    /**
     * @brief correct coulomb and non-coulomb energy and forces for linker connectivity elements
     *
     * @details with template parameter T, scaling of non-coulomb energy and forces can be applied to dihedrals
     *
     * @tparam T
     * @param coulombPotential
     * @param nonCoulombPotential
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
    double correctLinker(const potential::CoulombPotential &coulombPotential,
                         potential::NonCoulombPotential    &nonCoulombPotential,
                         physicalData::PhysicalData        &physicalData,
                         const simulationBox::Molecule     *molecule1,
                         const simulationBox::Molecule     *molecule2,
                         const size_t                       atomIndex1,
                         const size_t                       atomIndex2,
                         const double                       distance)
    {
        const auto chargeProduct = molecule1->getPartialCharge(atomIndex1) * molecule2->getPartialCharge(atomIndex2);

        auto [coulombEnergy, coulombForce] = coulombPotential.calculate(distance, chargeProduct);

        if constexpr (std::is_same_v<T, DihedralForceField>)
        {
            coulombEnergy *= (1.0 - settings::PotentialSettings::getScale14Coulomb());
            coulombForce  *= (1.0 - settings::PotentialSettings::getScale14Coulomb());
        }

        auto forceMagnitude = -coulombForce;
        physicalData.addCoulombEnergy(-coulombEnergy);

        const auto molType1  = molecule1->getMoltype();
        const auto molType2  = molecule2->getMoltype();
        const auto atomType1 = molecule1->getAtomType(atomIndex1);
        const auto atomType2 = molecule2->getAtomType(atomIndex2);
        const auto vdwType1  = molecule1->getInternalGlobalVDWType(atomIndex1);
        const auto vdwType2  = molecule2->getInternalGlobalVDWType(atomIndex2);

        const auto combinedIndices = {molType1, molType2, atomType1, atomType2, vdwType1, vdwType2};

        if (const auto nonCoulombPair = nonCoulombPotential.getNonCoulombPair(combinedIndices);
            distance < nonCoulombPair->getRadialCutOff())
        {
            auto [nonCoulombEnergy, nonCoulombForce] = nonCoulombPair->calculateEnergyAndForce(distance);

            if constexpr (std::is_same_v<T, DihedralForceField>)
            {
                nonCoulombEnergy *= (1.0 - settings::PotentialSettings::getScale14VanDerWaals());
                nonCoulombForce  *= (1.0 - settings::PotentialSettings::getScale14VanDerWaals());
            }

            forceMagnitude -= nonCoulombForce;
            physicalData.addNonCoulombEnergy(-nonCoulombEnergy);
        }

        return forceMagnitude;
    }
}   // namespace forceField

#endif   // _FORCE_FIELD_HPP_