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

#include "bondSection.hpp"

#include <format>   // for format
#include <string>   // for stoul, string, operator==, char_traits
#include <vector>   // for vector

#include "bondForceField.hpp"    // for BondForceField
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for TopologyException
#include "forceFieldClass.hpp"   // for ForceField
#include "simulationBox.hpp"     // for SimulationBox

using namespace input::topology;
using namespace simulationBox;
using namespace forceField;
using namespace customException;
using namespace engine;

/**
 * @brief processes the bond section of the topology file
 *
 * @details one line consists of 3 or 4 elements:
 * 1. atom index 1
 * 2. atom index 2
 * 3. bond type
 * 4. linker marked with a '*'
 *
 * @param line
 * @param engine
 *
 * @throws TopologyException if number of elements in line is
 * not 3 or 4
 * @throws TopologyException if atom indices are the same
 * (=same atoms)
 * @throws TopologyException if forth element is not a '*'
 */
void BondSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 3 && lineElements.size() != 4)
        throw TopologyException(
            std::format(
                "Wrong number of arguments in topology file bond section at "
                "line "
                "{} - number of elements has to be 3 or 4!",
                _lineNumber
            )
        );

    const auto atom1    = stoul(lineElements[0]);
    const auto atom2    = stoul(lineElements[1]);
    const auto bondType = stoul(lineElements[2]);
    auto       isLinker = false;

    if (4 == lineElements.size())
    {
        if (lineElements[3] == "*")
            isLinker = true;

        else
            throw TopologyException(
                std::format(
                    "Forth entry in topology file in bond section has to be a "
                    "\'*\' or empty at line {}!",
                    _lineNumber
                )
            );
    }

    if (atom1 == atom2)
        throw TopologyException(
            std::format(
                "Topology file shake section at line {} - atoms cannot be the "
                "same!",
                _lineNumber
            )
        );

    auto &simBox = engine.getSimulationBox();

    const auto [mol1, atomIdx1] = simBox.findMoleculeByAtomIndex(atom1);
    const auto [mol2, atomIdx2] = simBox.findMoleculeByAtomIndex(atom2);

    auto bondFF = BondForceField(mol1, mol2, atomIdx1, atomIdx2, bondType);
    bondFF.setIsLinker(isLinker);

    engine.getForceField().addBond(bondFF);
}

/**
 * @brief returns the keyword of the bond section
 *
 * @return "bonds"
 */
std::string BondSection::keyword() { return "bonds"; }

/**
 * @brief checks if bond sections ends normally
 *
 * @param endedNormal
 *
 * @throws TopologyException if endedNormal is false
 */
void BondSection::endedNormally(const bool endedNormal) const
{
    if (!endedNormal)
        throw TopologyException(
            std::format(
                "Topology file bond section at line {} - no end of section "
                "found!",
                _lineNumber
            )
        );
}