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

#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS, EXPECT_EQ

#include <string>   // for allocator, basic_string

#include "celllist.hpp"        // for CellList
#include "celllistSetup.hpp"   // for CellListSetup, setupCellList, setup
#include "engine.hpp"          // for Engine
#include "gtest/gtest.h"       // for Message, TestPartResult
#include "potential.hpp"       // for PotentialBruteForce, PotentialCellList
#include "testSetup.hpp"       // for TestSetup

#ifdef __PQ_LEGACY__
#include "potentialBruteForce.hpp"   // for PotentialBruteForce
#include "potentialCellList.hpp"     // for PotentialCellList
#endif

using namespace setup;

/**
 * @brief test the setup cell list function after parsing input
 *
 */
TEST_F(TestSetup, setupCellList)
{
    CellListSetup cellListSetup(*_engine);
    cellListSetup.setup();

#ifdef __PQ_LEGACY__
    EXPECT_EQ(
        typeid((_engine->getPotential())),
        typeid(potential::PotentialBruteForce)
    );
#else
    EXPECT_EQ(typeid((_engine->getPotential())), typeid(potential::Potential));
#endif

    _engine->getCellList().activate();
    cellListSetup.setup();

#ifdef __PQ_LEGACY__
    EXPECT_EQ(
        typeid((_engine->getPotential())),
        typeid(potential::PotentialCellList)
    );
#else
    EXPECT_EQ(typeid((_engine->getPotential())), typeid(potential::Potential));
#endif

    EXPECT_NO_THROW(setupCellList(*_engine));
}