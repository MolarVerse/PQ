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

TEST(TestReferencesOutput, writeReferencesFileEmitsHeaderAndBibtexBanner)
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

TEST(TestReferencesOutput, rejectsUnwritableOutput)
{
    OutputFileSettings::setRefFileName(".");

    EXPECT_THROW(ReferencesOutput::writeReferencesFile(), std::runtime_error);
}

TEST(TestReferencesOutput, rejectsMissingReferenceFiles)
{
    EXPECT_NO_THROW(ReferencesOutput::addReferenceFile("nonexistent.ref"));
    EXPECT_NO_THROW(ReferencesOutput::addReferenceFile("nonexistent.ref"));

    const std::string path = "default.refs.test";
    OutputFileSettings::setRefFileName(path);

    EXPECT_THROW(ReferencesOutput::writeReferencesFile(), std::runtime_error);
    EXPECT_FALSE(std::ifstream(path).good());

    ReferencesOutput::_referenceFileNames.erase("nonexistent.ref");
    ReferencesOutput::_bibtexFileNames.erase("nonexistent.ref.bib");
}
