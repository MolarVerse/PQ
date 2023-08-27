#include "angleForceField.hpp"       // for AngleForceField
#include "engine.hpp"                // for Engine
#include "exceptions.hpp"            // for TopologyException
#include "forceField.hpp"            // for ForceField
#include "simulationBox.hpp"         // for SimulationBox
#include "testTopologySection.hpp"   // for TestTopologySection
#include "topologySection.hpp"       // for AngleSection

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

/**
 * @brief test angle section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionAngle)
{
    std::vector<std::string>          lineElements = {"2", "1", "3", "7"};
    readInput::topology::AngleSection angleSection;
    angleSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getAngles().size(), 1);
    EXPECT_EQ(_engine->getForceField().getAngles()[0].getMolecules()[0], &(_engine->getSimulationBox().getMolecules()[0]));
    EXPECT_EQ(_engine->getForceField().getAngles()[0].getMolecules()[1], &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getAngles()[0].getMolecules()[2], &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getAngles()[0].getAtomIndices()[0], 0);
    EXPECT_EQ(_engine->getForceField().getAngles()[0].getAtomIndices()[1], 0);
    EXPECT_EQ(_engine->getForceField().getAngles()[0].getAtomIndices()[2], 1);
    EXPECT_EQ(_engine->getForceField().getAngles()[0].getType(), 7);

    lineElements = {"1", "1", "2", "3"};
    EXPECT_THROW(angleSection.processSection(lineElements, *_engine), customException::TopologyException);

    lineElements = {"1", "2", "7"};
    EXPECT_THROW(angleSection.processSection(lineElements, *_engine), customException::TopologyException);
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyAngle)
{
    readInput::topology::AngleSection angleSection;
    EXPECT_THROW(angleSection.endedNormally(false), customException::TopologyException);
    EXPECT_NO_THROW(angleSection.endedNormally(true));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}