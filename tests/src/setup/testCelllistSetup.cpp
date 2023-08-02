#include "celllistSetup.hpp"
#include "constants.hpp"
#include "exceptions.hpp"
#include "testSetup.hpp"

using namespace setup;

/**
 * @brief test the setup cell list function after parsing input
 *
 */
TEST_F(TestSetup, setupCellList)
{
    CellListSetup cellListSetup(_engine);
    cellListSetup.setup();

    EXPECT_EQ(typeid((_engine.getPotential())), typeid(potential::PotentialBruteForce));

    _engine.getCellList().activate();
    cellListSetup.setup();

    EXPECT_EQ(typeid((_engine.getPotential())), typeid(potential::PotentialCellList));

    EXPECT_NO_THROW(setupCellList(_engine));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}