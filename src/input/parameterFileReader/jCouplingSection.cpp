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

#include <format>   // for format

#include "constants/conversionFactors.hpp"   // for _DEG_TO_RAD_
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"                    // for ParameterFileException
#include "forceFieldClass.hpp"               // for ForceField
#include "jCouplingType.hpp"                 // for JCouplingType

using namespace input::parameterFile;
using namespace engine;
using namespace customException;
using namespace forceField;
using namespace constants;

/**
 * @brief returns the keyword of the j-coupling section
 *
 * @return "j-couplings"
 */
std::string JCouplingSection::keyword() { return "j-couplings"; }

/**
 * @brief processes one line of the j-coupling section of the parameter file and
 * adds the j-coupling type to the force field
 *
 * @details The line is expected to have the following format:
 * 1. jCouplingTypeId
 * 2. J0
 * 3. forceConstant
 * 4. a
 * 5. b
 * 6. c
 * 7. symmetry +/-/0 or anything
 *
 * According to the following equation:
 *
 * J = a * cos(phi + phaseShift)^2 + b * cos(phi + phaseShift) + c
 *
 * where phi is the dihedral angle
 *
 * V_J = forceConstant * (J - J_0)^2
 *
 * The symmetry parameter is used to determine the symmetry of the j-coupling
 *     - if symmetry is 0, no j-coupling is calculated
 *     - if symmetry is +, the j-coupling is calculated if J > J_0
 *     - if symmetry is -, the j-coupling is calculated if J < J_0
 *     - if symmetry is anything else, the j-coupling is calculated if J != J_0
 *
 * @param line
 * @param engine
 *
 * @throw ParameterFileException if number of elements in line
 * is not 7 or 8
 */
void JCouplingSection::processSection(
    std::vector<std::string> &lineElements,
    Engine                   &engine
)
{
    if (lineElements.size() != 6 && lineElements.size() != 7)
        throw ParameterFileException(std::format(
            "Wrong number of arguments in parameter file j-coupling section at "
            "line {} - number of elements has to be 7 or 8!",
            _lineNumber
        ));

    auto id            = stoul(lineElements[0]);
    auto J0            = stod(lineElements[1]);
    auto forceConstant = stod(lineElements[2]);
    auto a             = stod(lineElements[3]);
    auto b             = stod(lineElements[4]);
    auto c             = stod(lineElements[5]);

    auto upperSymmetry = true;
    auto lowerSymmetry = true;

    if (lineElements.size() == 8)
    {
        const auto &symmetry = lineElements[6];

        if (symmetry == "0")
            upperSymmetry = lowerSymmetry = false;

        if (symmetry == "+")
            lowerSymmetry = false;

        if (symmetry == "-")
            upperSymmetry = false;
    }

    auto jCouplingType = JCouplingType(id, J0, forceConstant, a, b, c);

    jCouplingType.setUpperSymmetry(upperSymmetry);
    jCouplingType.setLowerSymmetry(lowerSymmetry);

    engine.getForceField().addJCouplingType(jCouplingType);
}