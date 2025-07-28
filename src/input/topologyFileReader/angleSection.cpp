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

#include "angleSection.hpp"

#include <cstddef>   // for size_t
#include <format>    // for format
#include <string>    // for stoul, string, operator==, char_traits
#include <vector>    // for vector

#include "angleForceField.hpp"   // for AngleForceField
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
 * @brief processes the angle section of the topology file
 *
 * @details one line consists of 4 or 5 elements:
 * 1. atom index 1
 * 2. atom index 2 (center atom)
 * 3. atom index 3
 * 4. angle type
 * 5. linker marked with a '*' (optional)
 *
 * @param line
 * @param engine
 *
 * @throws TopologyException if number of elements in line is
 * not 4 or 5
 * @throws TopologyException if atom indices are the same
 * (=same atoms)
 * @throws TopologyException if fifth element is not a '*'
 */
void AngleSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 4 && lineElements.size() != 5)
        throw TopologyException(
            std::format(
                "Wrong number of arguments in topology file angle section at "
                "line "
                "{} - number of elements has to be 4 or 5!",
                _lineNumber
            )
        );

    auto atom1     = stoul(lineElements[0]);
    auto atom2     = stoul(lineElements[1]);
    auto atom3     = stoul(lineElements[2]);
    auto angleType = stoul(lineElements[3]);
    auto isLinker  = false;

    if (5 == lineElements.size())
    {
        if (lineElements[4] == "*")
            isLinker = true;

        else
            throw TopologyException(
                std::format(
                    "Fifth entry in topology file in angle section has to be a "
                    "\'*\' or empty at line {}!",
                    _lineNumber
                )
            );
    }

    if (atom1 == atom2 || atom1 == atom3 || atom2 == atom3)
        throw TopologyException(
            std::format(
                "Topology file angle section at line {} - atoms cannot be the "
                "same!",
                _lineNumber
            )
        );

    auto &simBox = engine.getSimulationBox();

    const auto [molecule1, atomIdx1] = simBox.findMoleculeByAtomIndex(atom1);
    const auto [molecule2, atomIdx2] = simBox.findMoleculeByAtomIndex(atom2);
    const auto [molecule3, atomIdx3] = simBox.findMoleculeByAtomIndex(atom3);

    const auto mols = std::vector<Molecule *>{molecule2, molecule1, molecule3};
    const auto atomIndices = std::vector<size_t>{atomIdx2, atomIdx1, atomIdx3};

    auto angleForceField = AngleForceField(mols, atomIndices, angleType);
    angleForceField.setIsLinker(isLinker);

    engine.getForceField().addAngle(angleForceField);
}

/**
 * @brief returns the keyword of the angle section
 *
 * @return "angles"
 */
std::string AngleSection::keyword() { return "angles"; }

/**
 * @brief checks if angle sections ends normally
 *
 * @param endedNormal
 *
 * @throws TopologyException if endedNormal is false
 */
void AngleSection::endedNormally(const bool endedNormal) const
{
    if (!endedNormal)
        throw TopologyException(
            std::format(
                "Topology file angle section at line {} - no end of section "
                "found!",
                _lineNumber
            )
        );
}