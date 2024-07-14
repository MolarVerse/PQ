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
using namespace simulationBox;
using namespace forceField;
using namespace customException;
using namespace engine;

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
 * @throws TopologyException if number of elements in line is
 * not 5 or 6
 * @throws TopologyException if atom indices are the same
 * (=same atoms)
 * @throws TopologyException if sixth element is not a '*'
 */
void DihedralSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw TopologyException(std::format(
            "Wrong number of arguments in topology file dihedral section at "
            "line {} - number of elements has to be 5 or 6!",
            _lineNumber
        ));

    const auto atom1        = stoul(lineElements[0]);
    const auto atom2        = stoul(lineElements[1]);
    const auto atom3        = stoul(lineElements[2]);
    const auto atom4        = stoul(lineElements[3]);
    const auto dihedralType = stoul(lineElements[4]);
    auto isLinker     = false;

    if (6 == lineElements.size())
    {
        if (lineElements[5] == "*")
            isLinker = true;

        else
            throw TopologyException(std::format(
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
        throw TopologyException(std::format(
            "Topology file dihedral section at line {} - atoms cannot be the "
            "same!",
            _lineNumber
        ));

    auto &simBox = engine.getSimulationBox();

    const auto [mol1, idx1] = simBox.findMoleculeByAtomIndex(atom1);
    const auto [mol2, idx2] = simBox.findMoleculeByAtomIndex(atom2);
    const auto [mol3, idx3] = simBox.findMoleculeByAtomIndex(atom3);
    const auto [mol4, idx4] = simBox.findMoleculeByAtomIndex(atom4);

    const auto mols     = std::vector<Molecule *>{mol1, mol2, mol3, mol4};
    const auto atomIdxs = std::vector<size_t>{idx1, idx2, idx3, idx4};

    auto dihedralFF = DihedralForceField(mols, atomIdxs, dihedralType);
    dihedralFF.setIsLinker(isLinker);

    engine.getForceField().addDihedral(dihedralFF);
}

/**
 * @brief returns the keyword of the dihedral section
 *
 * @return "dihedrals"
 */
std::string DihedralSection::keyword() { return "dihedrals"; }

/**
 * @brief checks if dihedral sections ends normally
 *
 * @param endedNormal
 *
 * @throws TopologyException if endedNormal is false
 */
void DihedralSection::endedNormally(const bool endedNormal) const
{
    if (!endedNormal)
        throw TopologyException(std::format(
            "Topology file dihedral section at line {} - no end of section "
            "found!",
            _lineNumber
        ));
}