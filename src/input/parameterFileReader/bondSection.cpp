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

#include "bondType.hpp"          // for BondType
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for ParameterFileException
#include "forceFieldClass.hpp"   // for ForceField

using namespace input::parameterFile;
using namespace customException;
using namespace engine;
using namespace forceField;

/**
 * @brief returns the keyword of the bond section
 *
 * @return "bonds"
 */
std::string BondSection::keyword() { return "bonds"; }

/**
 * @brief processes one line of the bond section of the parameter file and adds
 * the bond type to the force field
 *
 * @details The line is expected to have the following format:
 * 1. bondTypeId
 * 2. equilibriumDistance
 * 3. forceConstant
 *
 * @param line
 * @param engine
 *
 * @throw ParameterFileException if number of elements in line
 * is not 3
 * @throw ParameterFileException if equilibrium distance is
 * negative
 */
void BondSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 3)
        throw ParameterFileException(
            std::format(
                "Wrong number of arguments in parameter file bond section at "
                "line "
                "{} - number of elements has to be 3!",
                _lineNumber
            )
        );

    auto id                  = stoul(lineElements[0]);
    auto equilibriumDistance = stod(lineElements[1]);
    auto forceConstant       = stod(lineElements[2]);

    if (equilibriumDistance < 0.0)
        throw ParameterFileException(
            std::format(
                "Parameter file bond section at line {} - equilibrium distance "
                "has "
                "to be positive!",
                _lineNumber
            )
        );

    auto bondType = BondType(id, equilibriumDistance, forceConstant);

    engine.getForceField().addBondType(bondType);
}