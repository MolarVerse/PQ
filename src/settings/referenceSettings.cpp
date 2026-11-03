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

#include "referenceSettings.hpp"

/**
 * @brief Adds a reference file and its corresponding BibTeX file.
 *
 * @param referenceFileName The name of the reference file to add.
 */
void ReferenceSettings::addReferenceFile(const std::string& referenceFileName)
{
    _referenceFileNames.insert(referenceFileName);
    _bibtexFileNames.insert(referenceFileName + ".bib");
}

/**
 * @brief Retrieves the set of reference file names.
 *
 * @return A constant reference to the set of reference file names.
 */
const std::unordered_set<std::string>& ReferenceSettings::getReferenceFileNames(
) const
{
    return _referenceFileNames;
}

/**
 * @brief Retrieves the set of BibTeX file names.
 *
 * @return A constant reference to the set of BibTeX file names.
 */
const std::unordered_set<std::string>& ReferenceSettings::getBibtexFileNames(
) const
{
    return _bibtexFileNames;
}
