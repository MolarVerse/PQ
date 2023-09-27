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

#include "molecule.hpp"

#include "vector3d.hpp"

#include <algorithm>    // for std::ranges::for_each
#include <functional>   // for identity, equal_to
#include <iterator>     // for _Size, size
#include <ranges>       // for subrange

using namespace simulationBox;

/**
 * @brief finds number of different atom types in molecule
 *
 * @return int
 */
size_t Molecule::getNumberOfAtomTypes()
{
    std::vector<size_t> externalAtomTypes;

    std::ranges::transform(_atoms, std::back_inserter(externalAtomTypes), [](auto atom) { return atom->getExternalAtomType(); });

    return getNumberOfAtoms() - std::ranges::size(std::ranges::unique(externalAtomTypes));
}

/**
 * @brief calculates the center of mass of the molecule
 *
 * @details distances are calculated relative to the first atom
 *
 * @param box
 */
void Molecule::calculateCenterOfMass(const linearAlgebra::Vec3D &box)
{
    _centerOfMass            = linearAlgebra::Vec3D();
    const auto positionAtom1 = _atoms[0]->getPosition();

    // TODO: sonarlint until now not compatible with c++23
    //  auto f = [&_centerOfMass = _centerOfMass, &positionAtom1, &box = box](auto &&pair)
    //  {
    //      auto const &[mass, position]  = pair;
    //      _centerOfMass                += mass * (position - box * round((position - positionAtom1) / box));
    //  };
    //  std::ranges::for_each(std::ranges::views::zip(_masses, _positions), f);

    // TODO: change this loop to a range based for loop

    for (size_t i = 0; i < getNumberOfAtoms(); ++i)
    {
        const auto mass     = _atoms[i]->getMass();
        const auto position = _atoms[i]->getPosition();

        _centerOfMass += mass * (position - box * round((position - positionAtom1) / box));
    }

    _centerOfMass /= getMolMass();
}

/**
 * @brief scales the positions of the molecule by shifting the center of mass
 *
 * @param shiftFactors
 */
void Molecule::scale(const linearAlgebra::Vec3D &shiftFactors)
{
    const auto shift = _centerOfMass * (shiftFactors - 1.0);

    std::ranges::for_each(_atoms, [shift](auto atom) { atom->addPosition(shift); });
}