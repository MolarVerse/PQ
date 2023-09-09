#include "celllist.hpp"        // for CellList
#include "celllistSetup.hpp"   // for CellListSetup, setupCellList, setup
#include "engine.hpp"          // for Engine
#include "potential.hpp"       // for PotentialBruteForce, PotentialCellList
#include "testSetup.hpp"       // for TestSetup

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS, EXPECT_EQ
#include <string>          // for allocator, basic_string

using namespace setup;

/**
 * @brief test the setup cell list function after parsing input
 *
 */
TEST_F(TestSetup, setupCellList)
{
    CellListSetup cellListSetup(*_engine);
    cellListSetup.setup();

    EXPECT_EQ(typeid((_engine->getPotential())), typeid(potential::PotentialBruteForce));

    _engine->getCellList().activate();
    cellListSetup.setup();

    EXPECT_EQ(typeid((_engine->getPotential())), typeid(potential::PotentialCellList));

    EXPECT_NO_THROW(setupCellList(*_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}