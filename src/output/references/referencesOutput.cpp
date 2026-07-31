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

#include <filesystem>   // for is_directory, is_regular_file, path
#include <format>       // for format
#include <fstream>      // for fstream
#include <sstream>      // for ostringstream
#include <stdexcept>    // for runtime_error
#include <string>       // for string

#include "executablePath.hpp"       // for executablePath
#include "outputFileSettings.hpp"   // for OutputFileSettings
#include "references.hpp"           // for ReferencesOutput

using references::ReferencesOutput;
using namespace settings;

namespace
{
    std::filesystem::path referenceFilesPath()
    {
        const auto installedPath =
            utilities::installedDataPath("references");
        if (std::filesystem::is_directory(installedPath))
            return installedPath;

        const auto buildPath = std::filesystem::path(REFERENCES_PATH_);
        if (std::filesystem::is_directory(buildPath))
            return buildPath;

        throw std::runtime_error("PQ reference data could not be found");
    }

    void renderReferenceFile(
        const std::filesystem::path &path,
        std::ostream                &output
    )
    {
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error(
                std::format(
                    "PQ reference file \"{}\" could not be found",
                    path.string()
                )
            );

        std::ifstream referenceFile(path);
        if (!referenceFile.is_open())
            throw std::runtime_error(
                std::format(
                    "Could not open PQ reference file \"{}\"",
                    path.string()
                )
            );

        std::string line;
        while (getline(referenceFile, line)) output << line << '\n';

        if (referenceFile.bad())
            throw std::runtime_error(
                std::format(
                    "Could not read PQ reference file \"{}\"",
                    path.string()
                )
            );

        output << "\n\n";
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

    std::ostringstream rendered;

    // clang-format off
    rendered << "########################################################################\n";
    rendered << "#                                                                      #\n";
    rendered << "#  This file contains all references to the software and theory used.  #\n";
    rendered << "#                                                                      #\n";
    rendered << "########################################################################\n";
    rendered << '\n';
    // clang-format on

    renderReferenceFile(sourceDirectory / _PQ_FILE_, rendered);
    for (const auto &referenceFileName : _referenceFileNames)
        renderReferenceFile(sourceDirectory / referenceFileName, rendered);

    // clang-format off
    rendered << '\n';
    rendered << "########################################################################\n";
    rendered << "#                                                                      #\n";
    rendered << "#                            BIBTEX ENTRIES                            #\n";
    rendered << "#                                                                      #\n";
    rendered << "########################################################################\n";
    rendered << '\n';
    // clang-format on

    renderReferenceFile(
        sourceDirectory / (static_cast<std::string>(_PQ_FILE_) + ".bib"),
        rendered
    );
    for (const auto &referenceFileName : _bibtexFileNames)
        renderReferenceFile(sourceDirectory / referenceFileName, rendered);

    std::ofstream output(filename);
    if (!output.is_open())
        throw std::runtime_error(
            std::format("Could not open reference output file \"{}\"", filename)
        );

    output << rendered.str();
    output.close();
    if (!output)
        throw std::runtime_error(
            std::format(
                "Could not write reference output file \"{}\"",
                filename
            )
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
