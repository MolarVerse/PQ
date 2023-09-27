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

#include "bondSection.hpp"

#include "bondForceField.hpp"    // for BondForceField
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for TopologyException
#include "forceFieldClass.hpp"   // for ForceField
#include "simulationBox.hpp"     // for SimulationBox

#include <format>   // for format
#include <string>   // for stoul, string, operator==, char_traits
#include <vector>   // for vector

using namespace readInput::topology;

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
 * @throws customException::TopologyException if number of elements in line is not 3 or 4
 * @throws customException::TopologyException if atom indices are the same (=same atoms)
 * @throws customException::TopologyException if forth element is not a '*'
 */
void BondSection::processSection(std::vector<std::string> &lineElements, engine::Engine &engine)
{
    if (lineElements.size() != 3 && lineElements.size() != 4)
        throw customException::TopologyException(std::format(
            "Wrong number of arguments in topology file bond section at line {} - number of elements has to be 3 or 4!",
            _lineNumber));

    const auto atom1    = stoul(lineElements[0]);
    const auto atom2    = stoul(lineElements[1]);
    const auto bondType = stoul(lineElements[2]);
    auto       isLinker = false;

    if (4 == lineElements.size())
    {
        if (lineElements[3] == "*")
            isLinker = true;
        else
            throw customException::TopologyException(
                std::format("Forth entry in topology file in bond section has to be a \'*\' or empty at line {}!", _lineNumber));
    }

    if (atom1 == atom2)
        throw customException::TopologyException(
            std::format("Topology file shake section at line {} - atoms cannot be the same!", _lineNumber));

    const auto [molecule1, atomIndex1] = engine.getSimulationBox().findMoleculeByAtomIndex(atom1);
    const auto [molecule2, atomIndex2] = engine.getSimulationBox().findMoleculeByAtomIndex(atom2);

    auto bondForceField = forceField::BondForceField(molecule1, molecule2, atomIndex1, atomIndex2, bondType);
    bondForceField.setIsLinker(isLinker);

    engine.getForceField().addBond(bondForceField);
}

/**
 * @brief checks if bond sections ends normally
 *
 * @param endedNormal
 *
 * @throws customException::TopologyException if endedNormal is false
 */
void BondSection::endedNormally(bool endedNormal) const
{
    if (!endedNormal)
        throw customException::TopologyException(
            std::format("Topology file bond section at line {} - no end of section found!", _lineNumber));
}