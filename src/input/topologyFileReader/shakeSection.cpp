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

#include "shakeSection.hpp"

#include <format>   // for format

#include "bondConstraint.hpp"   // for BondConstraint
#include "constraints.hpp"      // for Constraints
#include "engine.hpp"           // for Engine
#include "exceptions.hpp"       // for TopologyException
#include "simulationBox.hpp"    // for SimulationBox

using namespace input::topology;
using namespace engine;
using namespace customException;
using namespace constraints;

/**
 * @brief processes the shake section of the topology file
 *
 * @details one line of the shake section contains 4 elements:
 * 1. atom index 1
 * 2. atom index 2
 * 3. target bond length
 * 4. linker (not used yet - not sure what it is for)
 *
 * @param line
 * @param engine
 *
 * @throws TopologyException if number of elements in line is
 * not 4
 * @throws TopologyException if atom indices are the same
 * (=same atoms)
 */
void ShakeSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 4 && lineElements.size() != 3)
        throw TopologyException(std::format(
            "Wrong number of arguments in topology file shake section at line "
            "{} - number of elements has to be 3 or 4!",
            _lineNumber
        ));

    const auto atom1      = stoul(lineElements[0]);
    const auto atom2      = stoul(lineElements[1]);
    const auto bondLength = stod(lineElements[2]);
    // TODO: auto linker = lineElements[3];

    if (atom1 == atom2)
        throw TopologyException(std::format(
            "Topology file shake section at line {} - atoms cannot be the "
            "same!",
            _lineNumber
        ));

    auto &simBox = engine.getSimulationBox();

    const auto [molecule1, atomIndex1] = simBox.findMoleculeByAtomIndex(atom1);
    const auto [molecule2, atomIndex2] = simBox.findMoleculeByAtomIndex(atom2);

    auto bondConstraint = BondConstraint(
        molecule1,
        molecule2,
        atomIndex1,
        atomIndex2,
        bondLength
    );

    engine.getConstraints().addBondConstraint(bondConstraint);
}

/**
 * @brief returns the keyword of the shake section
 *
 * @return std::string
 */
std::string ShakeSection::keyword() { return "shake"; }

/**
 * @brief checks if shake sections ends normally
 *
 * @param endedNormal
 *
 * @throws TopologyException if endedNormal is false
 */
void ShakeSection::endedNormally(const bool endedNormal) const
{
    if (!endedNormal)
        throw TopologyException(std::format(
            "Topology file shake section at line {} - no end of section found!",
            _lineNumber
        ));
}