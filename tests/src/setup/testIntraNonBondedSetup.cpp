#include "engine.hpp"                    // for Engine
#include "intraNonBonded.hpp"            // for IntraNonBonded
#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "intraNonBondedSetup.hpp"       // for setupIntraNonBonded
#include "molecule.hpp"                  // for Molecule
#include "simulationBox.hpp"             // for SimulationBox
#include "testSetup.hpp"                 // for TestSetup

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS
#include <vector>          // for vector, allocator

/**
 * @brief tests the setup of the intra non bonded interactions
 *
 */
TEST_F(TestSetup, setupIntraNonBonded)
{

    auto molecule                = simulationBox::Molecule(1);
    auto intraNonBondedContainer = intraNonBonded::IntraNonBondedContainer(1, {{-1}});

    _engine->getIntraNonBonded().addIntraNonBondedContainer(intraNonBondedContainer);
    _engine->getSimulationBox().addMolecule(molecule);

    _engine->getIntraNonBonded().deactivate();
    setup::setupIntraNonBonded(*_engine);

    EXPECT_EQ(_engine->getIntraNonBonded().getIntraNonBondedMaps().size(), 0);

    _engine->getIntraNonBonded().activate();
    setup::setupIntraNonBonded(*_engine);

    EXPECT_EQ(_engine->getIntraNonBonded().getIntraNonBondedMaps().size(), 1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}