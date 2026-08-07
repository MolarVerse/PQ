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
#include <sstream>
#include <stdexcept>
#include <string>

#include "outputFileSettings.hpp"
#include "referencesOutput.hpp"

using references::ReferencesOutput;
using namespace settings;

namespace
{
    std::string slurp(const std::string &path)
    {
        std::ifstream     in(path);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
}   // namespace

class ReferencesOutputTest : public ::testing::Test
{
   protected:
    static void removeReferenceFile(const std::string &path)
    {
        ReferencesOutput::_referenceFileNames.erase(path);
        ReferencesOutput::_bibtexFileNames.erase(path + ".bib");
    }
};

TEST_F(ReferencesOutputTest, writeReferencesFileEmitsHeaderAndBibtexBanner)
{
    const std::string path = "default.refs.test";
    OutputFileSettings::setRefFileName(path);

    ReferencesOutput::writeReferencesFile();

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

    ::remove(path.c_str());
}

TEST_F(ReferencesOutputTest, rejectsUnwritableOutput)
{
    OutputFileSettings::setRefFileName(".");

    EXPECT_THROW(ReferencesOutput::writeReferencesFile(), std::runtime_error);
}

#if defined(__linux__)
TEST_F(ReferencesOutputTest, rejectsFailedOutputWrites)
{
    OutputFileSettings::setRefFileName("/dev/full");

    EXPECT_THROW(ReferencesOutput::writeReferencesFile(), std::runtime_error);
}
#endif

TEST_F(ReferencesOutputTest, rendersAdditionalReferenceFiles)
{
    const auto referencePath =
        std::filesystem::absolute("additional-reference.ref.test");
    const auto bibtexPath =
        std::filesystem::path(referencePath.string() + ".bib");
    const std::string outputPath = "default.refs.test";

    std::ofstream(referencePath) << "ADDITIONAL REFERENCE\n";
    std::ofstream(bibtexPath) << "ADDITIONAL BIBTEX\n";
    ReferencesOutput::addReferenceFile(referencePath.string());
    OutputFileSettings::setRefFileName(outputPath);

    EXPECT_NO_THROW(ReferencesOutput::writeReferencesFile());
    const auto content = slurp(outputPath);
    EXPECT_NE(content.find("ADDITIONAL REFERENCE"), std::string::npos);
    EXPECT_NE(content.find("ADDITIONAL BIBTEX"), std::string::npos);

    removeReferenceFile(referencePath.string());
    std::filesystem::remove(referencePath);
    std::filesystem::remove(bibtexPath);
    std::filesystem::remove(outputPath);
}

#if !defined(_WIN32)
TEST_F(ReferencesOutputTest, rejectsUnreadableReferenceFiles)
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

    ReferencesOutput::addReferenceFile(unreadablePath.string());
    OutputFileSettings::setRefFileName(outputPath);
    EXPECT_THROW(ReferencesOutput::writeReferencesFile(), std::runtime_error);
    removeReferenceFile(unreadablePath.string());

    std::filesystem::permissions(
        unreadablePath,
        std::filesystem::perms::owner_all
    );
    std::filesystem::remove(unreadablePath);
}
#endif

TEST_F(ReferencesOutputTest, rejectsMissingReferenceFiles)
{
    const std::string outputPath = "default.refs.test";

    EXPECT_NO_THROW(ReferencesOutput::addReferenceFile("nonexistent.ref"));
    EXPECT_NO_THROW(ReferencesOutput::addReferenceFile("nonexistent.ref"));

    OutputFileSettings::setRefFileName(outputPath);

    EXPECT_THROW(ReferencesOutput::writeReferencesFile(), std::runtime_error);
    EXPECT_FALSE(std::ifstream(outputPath).good());

    removeReferenceFile("nonexistent.ref");
}
