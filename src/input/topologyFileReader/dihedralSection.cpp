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

#include "dihedralSection.hpp"

#include <algorithm>   // for sort, unique
#include <format>      // for format
#include <string>      // for string, allocator
#include <vector>      // for vector

#include "dihedralForceField.hpp"   // for BondForceField
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for TopologyException

using namespace input::topology;

/**
 * @brief processes the dihedral section of the topology file
 *
 * @details one line consists of 5 or 6 elements:
 * 1. atom index 1
 * 2. atom index 2
 * 3. atom index 3
 * 4. atom index 4
 * 5. dihedral type
 * 6. linker marked with a '*' (optional)
 *
 * @param line
 * @param engine
 *
 * @throws customException::TopologyException if number of elements in line is
 * not 5 or 6
 * @throws customException::TopologyException if atom indices are the same
 * (=same atoms)
 * @throws customException::TopologyException if sixth element is not a '*'
 */
void DihedralSection::processSection(
    std::vector<std::string> &lineElements,
    engine::Engine           &engine
)
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw customException::TopologyException(std::format(
            "Wrong number of arguments in topology file dihedral section at "
            "line {} - number of elements has to be 5 or 6!",
            _lineNumber
        ));

    auto atom1        = stoul(lineElements[0]);
    auto atom2        = stoul(lineElements[1]);
    auto atom3        = stoul(lineElements[2]);
    auto atom4        = stoul(lineElements[3]);
    auto dihedralType = stoul(lineElements[4]);
    auto isLinker     = false;

    if (6 == lineElements.size())
    {
        if (lineElements[5] == "*")
            isLinker = true;
        else
            throw customException::TopologyException(std::format(
                "Sixth entry in topology file in dihedral section has to be a "
                "\'*\' or empty at line {}!",
                _lineNumber
            ));
    }

    auto atoms = std::vector{atom1, atom2, atom3, atom4};
    std::ranges::sort(atoms);
    const auto [it, end] = std::ranges::unique(atoms);
    atoms.erase(it, end);
    if (4 != atoms.size())
        throw customException::TopologyException(std::format(
            "Topology file dihedral section at line {} - atoms cannot be the "
            "same!",
            _lineNumber
        ));

    const auto [molecule1, atomIndex1] =
        engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    const auto [molecule2, atomIndex2] =
        engine.getSimulationBox().findMoleculeByAtomIndex(atom2);
    const auto [molecule3, atomIndex3] =
        engine.getSimulationBox().findMoleculeByAtomIndex(atom3);
    const auto [molecule4, atomIndex4] =
        engine.getSimulationBox().findMoleculeByAtomIndex(atom4);

    auto dihedralForceField = forceField::DihedralForceField(
        {molecule1, molecule2, molecule3, molecule4},
        {atomIndex1, atomIndex2, atomIndex3, atomIndex4},
        dihedralType
    );
    dihedralForceField.setIsLinker(isLinker);

    engine.getForceField().addDihedral(dihedralForceField);
}

/**
 * @brief checks if dihedral sections ends normally
 *
 * @param endedNormal
 *
 * @throws customException::TopologyException if endedNormal is false
 */
void DihedralSection::endedNormally(const bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException(std::format(
            "Topology file dihedral section at line {} - no end of section "
            "found!",
            _lineNumber
        ));
}