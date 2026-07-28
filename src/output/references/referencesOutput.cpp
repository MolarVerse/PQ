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
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "executablePath.hpp"
#include "references.hpp"           // for ReferencesOutput
#include "outputFileSettings.hpp"   // for OutputFileSettings

using references::ReferencesOutput;
using namespace settings;

namespace
{
    std::filesystem::path referenceFilesPath()
    {
        const auto executable = utilities::executablePath();
        if (!executable.empty())
        {
            const auto installedPath =
                executable.parent_path().parent_path() / "share" / "PQ" /
                "references";
            if (std::filesystem::is_directory(installedPath))
                return installedPath;
        }

        const auto buildPath = std::filesystem::path(REFERENCES_PATH_);
        if (std::filesystem::is_directory(buildPath))
            return buildPath;

        throw std::runtime_error("PQ reference data could not be found");
    }
}   // namespace

/**
 * @brief writes the references file
 *
 * @param filename
 */
void ReferencesOutput::writeReferencesFile()
{
    const auto sourceDirectory = referenceFilesPath();
    const auto filename        = OutputFileSettings::getRefFileName();

    auto referenceFileNames = std::vector<std::string>{_PQ_FILE_};
    referenceFileNames.insert(
        referenceFileNames.end(),
        _referenceFileNames.begin(),
        _referenceFileNames.end()
    );
    referenceFileNames.emplace_back(
        static_cast<std::string>(_PQ_FILE_) + ".bib"
    );
    referenceFileNames.insert(
        referenceFileNames.end(),
        _bibtexFileNames.begin(),
        _bibtexFileNames.end()
    );

    for (const auto &referenceFileName : referenceFileNames)
    {
        const auto path = sourceDirectory / referenceFileName;
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error(
                std::format(
                    "PQ reference file \"{}\" could not be found",
                    path.string()
                )
            );
    }

    std::ofstream fp(filename);
    if (!fp.is_open())
        throw std::runtime_error(
            std::format("Could not open reference output file \"{}\"", filename)
        );

    auto printReference =
        [&fp, &sourceDirectory](const std::string &referenceFileName)
    {
        std::ifstream referenceFile(sourceDirectory / referenceFileName);

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
    if (!fp)
        throw std::runtime_error(
            std::format("Could not write reference output file \"{}\"", filename)
        );
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
