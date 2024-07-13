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

#include "nonCoulombicsSection.hpp"

#include <format>   // for format
#include <memory>   // for make_shared
#include <string>   // for stod, stoul, string, operator==

#include "buckinghamPair.hpp"         // for BuckinghamPair
#include "engine.hpp"                 // for Engine
#include "exceptions.hpp"             // for ParameterFileException
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "lennardJonesPair.hpp"       // for LennardJonesPair
#include "morsePair.hpp"              // for MorsePair
#include "nonCoulombPotential.hpp"    // for NonCoulombPotential
#include "potential.hpp"              // for Potential
#include "potentialSettings.hpp"      // for PotentialSettings
#include "stringUtilities.hpp"        // for toLowerCopy

using namespace input::parameterFile;
using namespace customException;
using namespace engine;
using namespace potential;
using namespace settings;
using namespace utilities;

/**
 * @brief keyword for nonCoulombics section
 *
 * @return "noncoulombics"
 */
std::string NonCoulombicsSection::keyword() { return "noncoulombics"; }

/**
 * @brief processes the nonCoulombics header of the parameter file
 *
 * @note type of forceField can be given as second argument
 *       default is lj (Lennard Jones) which overrides default of guff
 *
 * @param line
 * @param engine
 *
 * @throw ParameterFileException if type of nonCoulombic is not
 * lj, buckingham or morse
 */
void NonCoulombicsSection::processHeader(pq::strings &lineElements, Engine &)
{
    using enum NonCoulombType;

    if (lineElements.size() > 1)
    {
        const auto type = toLowerCopy(lineElements[1]);

        if (type == "lj")
            PotentialSettings::setNonCoulombType(LJ);

        else if (type == "buckingham")
            PotentialSettings::setNonCoulombType(BUCKINGHAM);

        else if (type == "morse")
            PotentialSettings::setNonCoulombType(MORSE);

        else
            throw ParameterFileException(std::format(
                "Invalid type of nonCoulombic in parameter file nonCoulombic "
                "section at line {} - has to be lj, buckingham or morse!",
                _lineNumber
            ));
    }
    else
        // default of guff gets overriden
        PotentialSettings::setNonCoulombType(LJ);
}

/**
 * @brief determines which nonCoulombic type is processed
 *
 * @param line
 * @param engine
 *
 * @throw ParameterFileException if nonCoulombic type is not
 * lj, buckingham or morse
 */
void NonCoulombicsSection::processSection(
    pq::strings &lineElements,
    Engine      &engine
)
{
    switch (PotentialSettings::getNonCoulombType())
    {
        using enum NonCoulombType;
        case LJ: processLJ(lineElements, engine); break;

        case BUCKINGHAM: processBuckingham(lineElements, engine); break;

        case MORSE: processMorse(lineElements, engine); break;

        default:
            throw ParameterFileException(std::format(
                "Wrong type of nonCoulombic in parameter file nonCoulombic "
                "section at line {}  - has to be lj, buckingham or morse!",
                _lineNumber
            ));
    }
}

/**
 * @brief processes the LJ nonCoulombics section of the parameter file and adds
 * the LJ pair to the nonCoulombic potential
 *
 * @details The line is expected to have the following format:
 * 1. global van der Waals type 1
 * 2. global van der Waals type 2
 * 3. c6
 * 4. c12
 * 5. cutOff (optional); if not given or -1, the global Coulomb radius cutOff is
 * used
 *
 * @param line
 * @param engine
 *
 * @throw ParameterFileException if number of elements in line
 * is not 4 or 5
 */
void NonCoulombicsSection::processLJ(pq::strings &lineElements, Engine &engine)
    const
{
    if (lineElements.size() != 4 && lineElements.size() != 5)
        throw ParameterFileException(std::format(
            "Wrong number of arguments in parameter file in Lennard Jones "
            "nonCoulombics section at line {} - number of "
            "elements has to be 4 or 5!",
            _lineNumber
        ));

    const size_t atomType1 = stoul(lineElements[0]);
    const size_t atomType2 = stoul(lineElements[1]);
    const auto   c6        = stod(lineElements[2]);
    const auto   c12       = stod(lineElements[3]);

    auto cutOff = 5 == lineElements.size() ? stod(lineElements[4]) : -1.0;

    const auto coulombCutOff = PotentialSettings::getCoulombRadiusCutOff();

    cutOff = cutOff < 0.0 ? coulombCutOff : cutOff;

    auto &pot       = engine.getPotential().getNonCoulombPotential();
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(pot);

    potential.addNonCoulombicPair(std::make_shared<LennardJonesPair>(
        atomType1,
        atomType2,
        cutOff,
        c6,
        c12
    ));
}

/**
 * @brief processes the buckingham nonCoulombics section of the parameter file
 * and adds the buckingham pair to the nonCoulombic potential
 *
 * @details The line is expected to have the following format:
 * 1. global van der Waals type 1
 * 2. global van der Waals type 2
 * 3. a
 * 4. dRho
 * 5. c6
 * 6. cutOff (optional); if not given or -1, the global Coulomb radius cutOff is
 * used
 *
 * @param line
 * @param engine
 *
 * @throw ParameterFileException if number of elements in line
 * is not 5 or 6
 */
void NonCoulombicsSection::processBuckingham(
    pq::strings &lineElements,
    Engine      &engine
) const
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw ParameterFileException(std::format(
            "Wrong number of arguments in parameter file in Lennard Jones "
            "nonCoulombics section at line {} - number of "
            "elements has to be 5 or 6!",
            _lineNumber
        ));

    const size_t atomType1 = stoul(lineElements[0]);
    const size_t atomType2 = stoul(lineElements[1]);
    const auto   a         = stod(lineElements[2]);
    const auto   dRho      = stod(lineElements[3]);
    const auto   c6        = stod(lineElements[4]);

    auto cutOff = 6 == lineElements.size() ? stod(lineElements[5]) : -1.0;

    const auto coulombCutOff = PotentialSettings::getCoulombRadiusCutOff();

    cutOff = cutOff < 0.0 ? coulombCutOff : cutOff;

    auto &pot       = engine.getPotential().getNonCoulombPotential();
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(pot);

    potential.addNonCoulombicPair(std::make_shared<BuckinghamPair>(
        atomType1,
        atomType2,
        cutOff,
        a,
        dRho,
        c6
    ));
}

/**
 * @brief processes the morse nonCoulombics section of the parameter file and
 * adds the morse pair to the nonCoulombic potential
 *
 * @details The line is expected to have the following format:
 * 1. global van der Waals type 1
 * 2. global van der Waals type 2
 * 3. dissociationEnergy
 * 4. wellWidth
 * 5. equilibriumDistance
 * 6. cutOff (optional); if not given or -1, the global Coulomb radius cutOff is
 * used
 *
 * @param line
 * @param engine
 *
 * @throw ParameterFileException if number of elements in line
 * is not 5 or 6
 */
void NonCoulombicsSection::processMorse(
    pq::strings &lineElements,
    Engine      &engine
) const
{
    if (lineElements.size() != 5 && lineElements.size() != 6)
        throw ParameterFileException(std::format(
            "Wrong number of arguments in parameter file in Lennard Jones "
            "nonCoulombics section at line {} - number of "
            "elements has to be 5 or 6!",
            _lineNumber
        ));

    const size_t atomType1           = stoul(lineElements[0]);
    const size_t atomType2           = stoul(lineElements[1]);
    const auto   dissociationEnergy  = stod(lineElements[2]);
    const auto   wellWidth           = stod(lineElements[3]);
    const auto   equilibriumDistance = stod(lineElements[4]);

    auto cutOff = 6 == lineElements.size() ? stod(lineElements[5]) : -1.0;

    const auto coulombCutOff = PotentialSettings::getCoulombRadiusCutOff();

    cutOff = cutOff < 0.0 ? coulombCutOff : cutOff;

    auto &pot       = engine.getPotential().getNonCoulombPotential();
    auto &potential = dynamic_cast<ForceFieldNonCoulomb &>(pot);

    potential.addNonCoulombicPair(std::make_shared<MorsePair>(
        atomType1,
        atomType2,
        cutOff,
        dissociationEnergy,
        wellWidth,
        equilibriumDistance
    ));
}