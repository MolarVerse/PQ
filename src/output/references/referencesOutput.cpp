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

#include "referencesOutput.hpp"

#include <algorithm>   // for for_each
#include <fstream>     // for fstream
#include <iostream>    // for cout
#include <string>      // for string

#include "references.hpp"           // for ReferencesOutput
#include "outputFileSettings.hpp"   // for OutputFileSettings

using references::ReferencesOutput;
using namespace settings;

/**
 * @brief writes the references file
 *
 * @param filename
 */
void ReferencesOutput::writeReferencesFile()
{
    const auto filename = OutputFileSettings::getRefFileName();

    std::ofstream fp(filename);

    auto printReference = [&fp](const std::string &referenceFileName)
    {
        const auto    filepath = _referenceFilesPath + "/" + referenceFileName;
        std::ifstream referenceFile(filepath);

        std::string line;
        while (getline(referenceFile, line)) fp << line << '\n';

        fp << "\n\n";
        referenceFile.close();
    };

    // clang-format off
    fp << "########################################################################\n";
    fp << "#                                                                      #\n";
    fp << "#  This file contains all references to the software and theory used.  #\n";
    fp << "#                                                                      #\n";
    fp << "########################################################################\n";
    fp << '\n';
    // clang-format on

    printReference(_PQ_FILE_);
    std::ranges::for_each(_referenceFileNames, printReference);

    // clang-format off
    fp << '\n';
    fp << "########################################################################\n";
    fp << "#                                                                      #\n";
    fp << "#                            BIBTEX ENTRIES                            #\n";
    fp << "#                                                                      #\n";
    fp << "########################################################################\n";
    fp << '\n';
    // clang-format on

    printReference(static_cast<std::string>(_PQ_FILE_) + ".bib");
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