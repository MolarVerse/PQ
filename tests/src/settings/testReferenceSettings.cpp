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

#include "referenceSettings.hpp"

TEST(TestReferenceSettings, NewlyConstructedHasNoReferenceOrBibtexFiles)
{
    ReferenceSettings settings;

    EXPECT_TRUE(settings.getReferenceFileNames().empty());
    EXPECT_TRUE(settings.getBibtexFileNames().empty());
}

TEST(TestReferenceSettings, AddReferenceFileAddsBothReferenceAndBibtexEntry)
{
    ReferenceSettings settings;

    settings.addReferenceFile(ReferenceFiles::PQ_FILE);

    EXPECT_TRUE(
        settings.getReferenceFileNames().contains(ReferenceFiles::PQ_FILE)
    );
    EXPECT_TRUE(settings.getBibtexFileNames().contains(
        std::string(ReferenceFiles::PQ_FILE) + ".bib"
    ));
}

TEST(TestReferenceSettings, AddReferenceFileIsIdempotent)
{
    ReferenceSettings settings;

    settings.addReferenceFile(ReferenceFiles::BERENDSEN_FILE);
    settings.addReferenceFile(ReferenceFiles::BERENDSEN_FILE);

    EXPECT_EQ(settings.getReferenceFileNames().size(), 1);
    EXPECT_EQ(settings.getBibtexFileNames().size(), 1);
}

TEST(TestReferenceSettings, AddReferenceFileAccumulatesDistinctFiles)
{
    ReferenceSettings settings;

    settings.addReferenceFile(ReferenceFiles::VELOCITY_VERLET_FILE);
    settings.addReferenceFile(ReferenceFiles::BERENDSEN_FILE);
    settings.addReferenceFile(ReferenceFiles::DFTBPLUS_FILE);

    const auto &referenceFiles = settings.getReferenceFileNames();
    EXPECT_EQ(referenceFiles.size(), 3);
    EXPECT_TRUE(referenceFiles.contains(ReferenceFiles::VELOCITY_VERLET_FILE));
    EXPECT_TRUE(referenceFiles.contains(ReferenceFiles::BERENDSEN_FILE));
    EXPECT_TRUE(referenceFiles.contains(ReferenceFiles::DFTBPLUS_FILE));

    const auto &bibtexFiles = settings.getBibtexFileNames();
    EXPECT_EQ(bibtexFiles.size(), 3);
    EXPECT_TRUE(bibtexFiles.contains(
        std::string(ReferenceFiles::VELOCITY_VERLET_FILE) + ".bib"
    ));
    EXPECT_TRUE(bibtexFiles.contains(
        std::string(ReferenceFiles::BERENDSEN_FILE) + ".bib"
    ));
    EXPECT_TRUE(bibtexFiles.contains(
        std::string(ReferenceFiles::DFTBPLUS_FILE) + ".bib"
    ));
}
