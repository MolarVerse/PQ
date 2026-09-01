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

#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "physicalData.hpp"          // for PhysicalData

namespace forceField
{
    template <typename T>
    double correctLinker(
        const potential::CoulombPotential &coulombPotential,
        potential::NonCoulombPotential    &nonCoulombPotential,
        physicalData::PhysicalData        &physicalData,
        const molsys::Molecule            *molecule1,
        const molsys::Molecule            *molecule2,
        const AtomIndex                    atomIndex1,
        const AtomIndex                    atomIndex2,
        const double                       distance
    );
}   // namespace forceField

#ifndef _FORCE_FIELD_TPP_
#include "forcefield.tpp.hpp"   // IWYU pragma: export - DO NOT MOVE THIS LINE
#endif

#endif   // _FORCE_FIELD_HPP_
