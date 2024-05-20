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

#include "jCouplingSection.hpp"

#include <algorithm>   // for sort, unique
#include <format>      // for format
#include <string>      // for string, allocator
#include <vector>      // for vector

#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for TopologyException
#include "jCouplingForceField.hpp"   // for JCouplingForceField

using namespace input::topology;

/**
 * @brief processes the j-coupling section of the topology file
 *
 * @details one line consists of 5 or 6 elements:
 * 1. atom index 1
 * 2. atom index 2
 * 3. atom index 3
 * 4. atom index 4
 * 5. j-coupling type
 *
 * @param line
 * @param engine
 *
 * @throws customException::TopologyException if number of elements in line
 is not 5 or 6
 * @throws customException::TopologyException if atom indices are the same
 (=same atoms)
 */
void JCouplingSection::processSection(
    std::vector<std::string> &lineElements,
    engine::Engine           &engine
)
{
    if (lineElements.size() != 5)
        throw customException::TopologyException(std::format(
            "Wrong number of arguments in topology file j-coupling "
            "section at line {} - number of elements has to be 5!",
            _lineNumber
        ));

    auto atom1        = stoul(lineElements[0]);
    auto atom2        = stoul(lineElements[1]);
    auto atom3        = stoul(lineElements[2]);
    auto atom4        = stoul(lineElements[3]);
    auto dihedralType = stoul(lineElements[4]);

    auto atoms = std::vector{atom1, atom2, atom3, atom4};
    std::ranges::sort(atoms);
    const auto [it, end] = std::ranges::unique(atoms);
    atoms.erase(it, end);

    if (4 != atoms.size())
        throw customException::TopologyException(std::format(
            "Topology file dihedral section at line {} "
            "- atoms cannot be the same!",
            _lineNumber
        ));

    auto simBox = engine.getSimulationBox();

    const auto [molecule1, atomIndex1] = simBox.findMoleculeByAtomIndex(atom1);
    const auto [molecule2, atomIndex2] = simBox.findMoleculeByAtomIndex(atom2);
    const auto [molecule3, atomIndex3] = simBox.findMoleculeByAtomIndex(atom3);
    const auto [molecule4, atomIndex4] = simBox.findMoleculeByAtomIndex(atom4);

    auto jCouplingForceField = forceField::JCouplingForceField(
        {molecule1, molecule2, molecule3, molecule4},
        {atomIndex1, atomIndex2, atomIndex3, atomIndex4},
        dihedralType
    );

    engine.getForceField().addJCoupling(jCouplingForceField);
}

/**
 * @brief checks if j-coupling section ends normally
 *
 * @param endedNormal
 *
 * @throws customException::TopologyException if endedNormal is false
 */
void JCouplingSection::endedNormally(const bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException(std::format(
            "Topology file j-coupling section at line {} "
            "- no end of section found!",
            _lineNumber
        ));
}