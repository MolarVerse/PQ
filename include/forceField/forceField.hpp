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

#ifndef _FORCE_FIELD_HPP_

#define _FORCE_FIELD_HPP_

#include <cstddef>   // for size_t
#include <memory>    // for __shared_ptr_access, shared_ptr

#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "physicalData.hpp"          // for PhysicalData
#include "potentialSettings.hpp"     // for PotentialSettings

namespace forceField
{
    template <typename T>
    double correctLinker(
        const potential::CoulombPotential &coulombPotential,
        potential::NonCoulombPotential    &nonCoulombPotential,
        physicalData::PhysicalData        &physicalData,
        const simulationBox::Molecule     *molecule1,
        const simulationBox::Molecule     *molecule2,
        const size_t                       atomIndex1,
        const size_t                       atomIndex2,
        const double                       distance
    );
}   // namespace forceField

#include "forcefield.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _FORCE_FIELD_HPP_