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

#include "referencesOutput.hpp"

#include "outputFileSettings.hpp"   // for OutputFileSettings

#include <algorithm>   // for for_each
#include <fstream>     // for fstream
#include <iostream>    // for cout
#include <string>      // for string

using references::ReferencesOutput;

/**
 * @brief writes the references file
 *
 * @param filename
 */
void ReferencesOutput::writeReferencesFile()
{
    const auto    filename = settings::OutputFileSettings::getReferenceFileName();
    std::ofstream fp(filename);

    auto printReference = [&fp](const std::string &referenceFileName)
    {
        std::ifstream referenceFile(_referenceFilesPath + "/" + referenceFileName);
        std::string   line;
        while (getline(referenceFile, line))
            fp << line << '\n';

        fp << "\n\n";
        referenceFile.close();
    };

    fp << "################################################################################\n";
    fp << "#                                                                              #\n";
    fp << "#  This file contains all references to the used software and the used theory  #\n";
    fp << "#                                                                              #\n";
    fp << "################################################################################\n";
    fp << '\n';

    std::ranges::for_each(_referenceFileNames, printReference);

    fp << '\n';
    fp << "################################################################################\n";
    fp << "#                                                                              #\n";
    fp << "#                               BIBTEX ENTIRES                                 #\n";
    fp << "#                                                                              #\n";
    fp << "################################################################################\n";
    fp << '\n';

    std::ranges::for_each(_bibtexFileNames, printReference);

    fp.close();
}

/**
 * @brief adds a reference file to the list of reference files and bibtex files
 *
 * @param referenceFileName
 */
void ReferencesOutput::addReferenceFile(const std::string &referenceFileName)
{
    _referenceFileNames.insert(referenceFileName);
    _bibtexFileNames.insert(referenceFileName + ".bib");
}