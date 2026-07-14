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

#include "inputValidation.hpp"

#include "exceptions.hpp"          // for InputFileException
#include "inputFileReader.hpp"     // for InputFileReader
#include "potentialSettings.hpp"   // for PotentialSettings

using namespace input;
using namespace settings;
using namespace customException;

/**
 * @brief validates semantic dependencies between parsed input keywords
 *
 * @details this is intended for cross-keyword checks that cannot be
 * verified within a single keyword parser.
 */
void input::validateInputConfiguration(const InputFileReader& inputFileReader)
{
    validateReactionFieldCoulomb(inputFileReader);
}

/**
 * @brief validates cross-keyword dependencies for the reaction field long range
 * coulomb correction
 *
 * @throws InputFileException if reaction-field Coulomb long-range correction
 * is selected but `rf_epsilon` is missing in the current input file
 */
void input::validateReactionFieldCoulomb(const InputFileReader& inputFileReader)
{
    using enum CoulombLongRangeType;

    const auto longRangeCorrection =
        PotentialSettings::getCoulombLongRangeType();

    if (longRangeCorrection == REACTION_FIELD &&
        !inputFileReader.getKeywordSet("rf_epsilon"))
        throw InputFileException(
            "Missing required keyword \"rf_epsilon\" in input file: it must "
            "be set when the Coulomb long-range correction is set to "
            "\"reaction-field\"."
        );
}
