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

#include <string>
#include <vector>

#include "exceptions.hpp"
#include "hybridSetup.hpp"
#include "settings.hpp"
#include "testSetup.hpp"

using namespace setup;
using namespace settings;
using namespace customException;

/* ---------- free function ---------- */

TEST_F(TestSetup, setupHybridIsNoOpWhenQMMMNotActive)
{
    Settings::setJobtype(JobType::MM_MD);   // not QMMM_MD
    EXPECT_NO_THROW(setupHybrid(*_engine));
}

/* ---------- parseSelectionNoPython ---------- */

TEST_F(TestSetup, parseSelectionNoPythonSingleIndex)
{
    HybridSetup hs(*_engine);
    const auto  v = hs.parseSelectionNoPython("3", "qm_center");
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], 3);
}

TEST_F(TestSetup, parseSelectionNoPythonCommaList)
{
    HybridSetup hs(*_engine);
    const auto  v = hs.parseSelectionNoPython("1,3,5", "qm_center");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 3);
    EXPECT_EQ(v[2], 5);
}

TEST_F(TestSetup, parseSelectionNoPythonRange)
{
    HybridSetup hs(*_engine);
    const auto  v = hs.parseSelectionNoPython("2-5", "qm_center");
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], 2);
    EXPECT_EQ(v[3], 5);
}

TEST_F(TestSetup, parseSelectionNoPythonMixedRangeAndList)
{
    HybridSetup hs(*_engine);
    const auto  v = hs.parseSelectionNoPython("1,3-4,7", "qm_center");
    ASSERT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 3);
    EXPECT_EQ(v[2], 4);
    EXPECT_EQ(v[3], 7);
}

TEST_F(TestSetup, parseSelectionNoPythonEmptyThrows)
{
    HybridSetup hs(*_engine);
    EXPECT_THROW(
        hs.parseSelectionNoPython("", "qm_center"),
        InputFileException
    );
}

/* ---------- parseSelection ---------- */

TEST_F(TestSetup, parseSelectionEmptyReturnsZeroOnly)
{
    HybridSetup hs(*_engine);
    const auto  v = hs.parseSelection("", "qm_center");
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], 0);
}

TEST_F(TestSetup, parseSelectionSortsAndDeduplicates)
{
    HybridSetup hs(*_engine);
    const auto  v = hs.parseSelection("5,1,3,1", "qm_center");
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 3);
    EXPECT_EQ(v[2], 5);
}

#ifndef PYTHON_ENABLED
TEST_F(TestSetup, parseSelectionWithLettersThrowsWithoutPython)
{
    HybridSetup hs(*_engine);
    EXPECT_THROW(
        hs.parseSelection("not_a_number", "qm_center"),
        InputFileException
    );
}
#endif

/* ---------- setup throws ---------- */

TEST_F(TestSetup, setupThrowsNotImplemented)
{
    HybridSetup hs(*_engine);
    EXPECT_THROW(hs.setup(), UserInputException);
}
