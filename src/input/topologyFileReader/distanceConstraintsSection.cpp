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

#include "distanceConstraintsSection.hpp"   // for DistanceConstraintsSection

#include <format>   // for format

#include "constraints.hpp"          // for Constraints
#include "distanceConstraint.hpp"   // for DistanceConstraint
#include "engine.hpp"               // for Engine
#include "exceptions.hpp"           // for TopologyException
#include "simulationBox.hpp"        // for SimulationBox

using namespace input::topology;
using namespace engine;
using namespace customException;
using namespace constraints;

/**
 * @brief processes the distance constraints section of the topology file
 *
 * @details one line of the distance constraints section contains 4 elements:
 * 1. atom index 1
 * 2. atom index 2
 * 3. lower distance
 * 4. upper distance
 * 5. spring constant
 * 6. dk/dt
 *
 * @param line
 * @param engine
 *
 * @throws TopologyException if number of elements in line is
 * not 4
 * @throws TopologyException if atom indices are the same
 * (=same atoms)
 * @throws TopologyException if lower distance is greater than
 * upper distance
 */
void DistanceConstraintsSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 6)
        throw TopologyException(
            std::format(
                "Wrong number of arguments in topology file \"Distance "
                "Constraints\" section at line {} - number of elements has to "
                "be "
                "6!",
                _lineNumber
            )
        );

    auto atom1             = stoul(lineElements[0]);
    auto atom2             = stoul(lineElements[1]);
    auto lowerDistance     = stod(lineElements[2]);
    auto upperDistance     = stod(lineElements[3]);
    auto springConstant    = stod(lineElements[4]);
    auto dSpringConstantDt = stod(lineElements[5]);

    if (atom1 == atom2)
        throw TopologyException(
            std::format(
                "Topology file \"Distance "
                "Constraints\" at line {} - atoms cannot be the same!",
                _lineNumber
            )
        );

    if (lowerDistance > upperDistance)
        throw TopologyException(
            std::format(
                "Topology file \"Distance "
                "Constraints\" at line {} - lower distance cannot be greater "
                "than upper distance!",
                _lineNumber
            )
        );

    auto &simBox = engine.getSimulationBox();

    const auto [molecule1, atomIndex1] = simBox.findMoleculeByAtomIndex(atom1);
    const auto [molecule2, atomIndex2] = simBox.findMoleculeByAtomIndex(atom2);

    auto distanceConstraint = DistanceConstraint(
        molecule1,
        molecule2,
        atomIndex1,
        atomIndex2,
        lowerDistance,
        upperDistance,
        springConstant,
        dSpringConstantDt
    );

    engine.getConstraints().addDistanceConstraint(distanceConstraint);
}

/**
 * @brief returns the keyword of the distance constraints section
 *
 * @return "dist_constraints"
 */
std::string DistanceConstraintsSection::keyword() { return "dist_constraints"; }

/**
 * @brief checks if distance constraints sections ends normally
 *
 * @param endedNormal
 *
 * @throws TopologyException if endedNormal is false
 */
void DistanceConstraintsSection::endedNormally(const bool endedNormal) const
{
    if (!endedNormal)
        throw TopologyException(
            std::format(
                "Topology file error in \"Distance Constraints\" section at "
                "line "
                "{} - no end of section found!",
                _lineNumber
            )
        );
}