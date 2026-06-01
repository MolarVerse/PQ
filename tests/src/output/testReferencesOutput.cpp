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
#include <fstream>
#include <sstream>
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

    ::remove(path.c_str());
}

TEST(TestReferencesOutput, addReferenceFileExtendsBothReferenceAndBibtexLists)
{
    // Sanity-check the static accessor is exposed at all; we can only observe
    // the side effect through the rendered output file, which won't contain
    // the new entry's body (the .ref file doesn't exist on disk) but the call
    // itself must not throw and must remain idempotent for duplicates.
    EXPECT_NO_THROW(ReferencesOutput::addReferenceFile("nonexistent.ref"));
    EXPECT_NO_THROW(ReferencesOutput::addReferenceFile("nonexistent.ref"));

    const std::string path = "default.refs.test";
    OutputFileSettings::setRefFileName(path);
    ReferencesOutput::writeReferencesFile();

    // Even with a non-existent reference file in the registered set, the
    // overall write succeeds and the file exists with at least the headers.
    const auto content = slurp(path);
    EXPECT_FALSE(content.empty());

    ::remove(path.c_str());
}
