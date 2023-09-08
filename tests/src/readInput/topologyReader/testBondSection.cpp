#include "bondForceField.hpp"        // for BondForceField
#include "bondSection.hpp"           // for BondSection
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for TopologyException
#include "forceFieldClass.hpp"       // for ForceField
#include "simulationBox.hpp"         // for SimulationBox
#include "testTopologySection.hpp"   // for TestTopologySection

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, EXPECT_THROW, TestInfo...
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

/**
 * @brief test bond section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionBond)
{
    std::vector<std::string>         lineElements = {"1", "2", "7"};
    readInput::topology::BondSection bondSection;
    bondSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getBonds().size(), 1);
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getMolecule1(), &(_engine->getSimulationBox().getMolecules()[0]));
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getMolecule2(), &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getAtomIndex1(), 0);
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getAtomIndex2(), 0);
    EXPECT_EQ(_engine->getForceField().getBonds()[0].getType(), 7);
    EXPECT_EQ(_engine->getForceField().getBonds()[0].isLinker(), false);

    lineElements = {"1", "2", "7", "*"};
    bondSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getBonds()[1].isLinker(), true);

    lineElements = {"1", "1", "7"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::TopologyException);

    lineElements = {"1", "2", "7", "1", "2"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::TopologyException);

    lineElements = {"1", "2", "7", "#"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::TopologyException);
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyBond)
{
    readInput::topology::BondSection bondSection;
    EXPECT_THROW(bondSection.endedNormally(false), customException::TopologyException);
    EXPECT_NO_THROW(bondSection.endedNormally(true));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}