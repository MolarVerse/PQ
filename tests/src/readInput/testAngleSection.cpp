#include "exceptions.hpp"
#include "forceField.hpp"
#include "testTopologySection.hpp"
#include "topologySection.hpp"

/**
 * @brief test angle section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionAngle)
{
    std::vector<std::string> lineElements = {"1", "2", "3", "7"};
    readInput::AngleSection  angleSection;
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
    readInput::AngleSection angleSection;
    EXPECT_THROW(angleSection.endedNormally(false), customException::TopologyException);
    EXPECT_NO_THROW(angleSection.endedNormally(true));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}