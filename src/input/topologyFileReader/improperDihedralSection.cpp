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

#include "improperDihedralSection.hpp"

#include <algorithm>   // for sort, unique
#include <format>      // for format
#include <string>      // for string, allocator
#include <vector>      // for vector

#include "dihedralForceField.hpp"   // for BondForceField
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for TopologyException

using namespace input::topology;
using namespace forceField;
using namespace customException;
using namespace engine;

/**
 * @brief processes the improper section of the topology file
 *
 * @details one line consists of 5 elements (cannot be linker!):
 * 1. atom index 1
 * 2. atom index 2
 * 3. atom index 3
 * 4. atom index 4
 * 5. improper dihedral type
 *
 * @param line
 * @param engine
 *
 * @throws TopologyException if number of elements in line is
 * not 5
 * @throws TopologyException if atom indices are the same
 * (=same atoms)
 */
void ImproperDihedralSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 5)
        throw TopologyException(
            std::format(
                "Wrong number of arguments in topology file improper dihedral "
                "section at "
                "line {} - number of elements has to be 5!",
                _lineNumber
            )
        );

    auto atom1                = stoul(lineElements[0]);
    auto atom2                = stoul(lineElements[1]);
    auto atom3                = stoul(lineElements[2]);
    auto atom4                = stoul(lineElements[3]);
    auto improperDihedralType = stoul(lineElements[4]);

    auto atoms = std::vector{atom1, atom2, atom3, atom4};
    std::ranges::sort(atoms);
    const auto [it, end] = std::ranges::unique(atoms);
    atoms.erase(it, end);

    if (4 != atoms.size())
        throw TopologyException(
            std::format(
                "Topology file improper dihedral section at line {} - atoms "
                "cannot "
                "be the same!",
                _lineNumber
            )
        );

    auto &simBox = engine.getSimulationBox();

    const auto [mol1, idx1] = simBox.findMoleculeByAtomIndex(atom1);
    const auto [mol2, idx2] = simBox.findMoleculeByAtomIndex(atom2);
    const auto [mol3, idx3] = simBox.findMoleculeByAtomIndex(atom3);
    const auto [mol4, idx4] = simBox.findMoleculeByAtomIndex(atom4);

    const auto mols     = std::vector{mol1, mol2, mol3, mol4};
    const auto atomIdxs = std::vector{idx1, idx2, idx3, idx4};

    auto improperFF = DihedralForceField(mols, atomIdxs, improperDihedralType);

    engine.getForceField().addImproperDihedral(improperFF);
}

/**
 * @brief returns the keyword of the improper dihedral section
 *
 * @return "impropers"
 */
std::string ImproperDihedralSection::keyword() { return "impropers"; }

/**
 * @brief checks if improper dihedral sections ends normally
 *
 * @param endedNormal
 *
 * @throws TopologyException if endedNormal is false
 */
void ImproperDihedralSection::endedNormally(const bool endedNormal) const
{
    if (!endedNormal)
        throw TopologyException(
            std::format(
                "Topology file improper dihedral section at line {} - no end "
                "of "
                "section found!",
                _lineNumber
            )
        );
}