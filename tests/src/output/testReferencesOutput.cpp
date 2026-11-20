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

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "referenceSettings.hpp"
#include "referencesOutput.hpp"
#include "testOutputBase.hpp"
#include "throwWithMessage.hpp"

TEST(ReferencesOutputTest, writeReferencesFileEmitsHeaderAndBibtexBanner)
{
    const std::string path = "default.refs.test";

    ReferenceSettings referenceSettings;
    references::ReferencesOutput(referenceSettings).writeReferencesFile(path);

    const auto content = slurp(path);
    // Top banner.
    EXPECT_NE(
        content.find(
            "This file contains all references to the software and theory used."
        ),
        std::string::npos
    );
    // Bibtex section banner.
    EXPECT_NE(content.find("BIBTEX ENTRIES"), std::string::npos);

    if (const char *marker = std::getenv("PQ_TEST_EXPECTED_REFERENCE_MARKER"))
    {
        EXPECT_NE(content.find(marker), std::string::npos);
    }

    const auto errorCode = std::remove(path.c_str());
    EXPECT_EQ(errorCode, 0) << "Failed to remove file: " << path;
}

TEST(ReferencesOutputTest, rejectsUnwritableOutput)
{
    ReferenceSettings referenceSettings;
    EXPECT_THROW_MSG(
        references::ReferencesOutput(referenceSettings)
            .writeReferencesFile("."),
        std::runtime_error,
        "Could not open reference output file \".\""

    );
}

#if defined(__linux__)
TEST(ReferencesOutputTest, rejectsFailedOutputWrites)
{
    ReferenceSettings referenceSettings;
    EXPECT_THROW_MSG(
        references::ReferencesOutput(referenceSettings)
            .writeReferencesFile("/dev/full"),
        std::runtime_error,
        "Could not write reference output file \"/dev/full\""

    );
}
#endif

TEST(ReferencesOutputTest, rendersAdditionalReferenceFiles)
{
    const auto referencePath =
        std::filesystem::absolute("additional-reference.ref.test");
    const auto bibtexPath =
        std::filesystem::path(referencePath.string() + ".bib");
    const std::string outputPath = "default.refs.test";

    std::ofstream(referencePath) << "ADDITIONAL REFERENCE\n";
    std::ofstream(bibtexPath) << "ADDITIONAL BIBTEX\n";
    ReferenceSettings referenceSettings;
    referenceSettings.addReferenceFile(referencePath.string());

    EXPECT_NO_THROW(
        references::ReferencesOutput(referenceSettings)
            .writeReferencesFile(outputPath)
    );
    const auto content = slurp(outputPath);
    EXPECT_NE(content.find("ADDITIONAL REFERENCE"), std::string::npos);
    EXPECT_NE(content.find("ADDITIONAL BIBTEX"), std::string::npos);

    std::filesystem::remove(referencePath);
    std::filesystem::remove(bibtexPath);
    std::filesystem::remove(outputPath);
}

#if !defined(_WIN32)
TEST(ReferencesOutputTest, rejectsUnreadableReferenceFiles)
{
    const auto unreadablePath =
        std::filesystem::absolute("unreadable-reference.ref.test");
    const std::string outputPath = "default.refs.test";
    std::ofstream(unreadablePath) << "UNREADABLE REFERENCE\n";
    std::filesystem::permissions(unreadablePath, std::filesystem::perms::none);

    if (std::ifstream(unreadablePath).is_open())
    {
        std::filesystem::permissions(
            unreadablePath,
            std::filesystem::perms::owner_all
        );
        std::filesystem::remove(unreadablePath);
        GTEST_SKIP() << "The current user can read files without permissions";
    }

    ReferenceSettings referenceSettings;
    referenceSettings.addReferenceFile(unreadablePath.string());

    EXPECT_THROW_MSG(
        references::ReferencesOutput(referenceSettings)
            .writeReferencesFile(outputPath),
        std::runtime_error,
        std::format(
            "Could not open PQ reference file "
            "\"{}\"",
            unreadablePath.string()
        )
    );

    std::filesystem::permissions(
        unreadablePath,
        std::filesystem::perms::owner_all
    );
    std::filesystem::remove(unreadablePath);
}
#endif

TEST(ReferencesOutputTest, rejectsMissingReferenceFiles)
{
    const std::string outputPath = "default.refs.test";

    std::string file = "nonexistent.ref";

    ReferenceSettings referenceSettings;
    EXPECT_NO_THROW(referenceSettings.addReferenceFile(file));
    EXPECT_NO_THROW(referenceSettings.addReferenceFile(file));

    EXPECT_THROW_MSG(
        references::ReferencesOutput(referenceSettings)
            .writeReferencesFile(outputPath),
        std::runtime_error,
        std::format(
            "PQ reference file "
            "\"{}/{}\" could not be found",
            _REFERENCES_PATH_,
            file
        )
    );
    EXPECT_FALSE(std::ifstream(outputPath).good());
}
