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

#ifndef FORCE_FIELD__FORCE_FIELD_HPP
#define FORCE_FIELD__FORCE_FIELD_HPP

#include "coulombPotential.hpp"
#include "molecule.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"

namespace ff
{
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
    );
}   // namespace ff

#ifndef FORCE_FIELD__FORCEFIELD_TPP
#include "forcefield.tpp"   // IWYU pragma: export
#endif

#endif   // FORCE_FIELD__FORCE_FIELD_HPP
