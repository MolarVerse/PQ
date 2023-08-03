#include "exceptions.hpp"
#include "forceField.hpp"
#include "testTopologySection.hpp"
#include "topologySection.hpp"

/**
 * @brief test impropers section processing one line
 *
 */
TEST_F(TestTopologySection, processSectionImproperDihedral)
{
    std::vector<std::string>                     lineElements = {"1", "2", "3", "4", "7"};
    readInput::topology::ImproperDihedralSection improperDihedralSection;
    improperDihedralSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals().size(), 1);
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getMolecules()[0],
              &(_engine->getSimulationBox().getMolecules()[0]));
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getMolecules()[1],
              &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getMolecules()[2],
              &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getMolecules()[3],
              &(_engine->getSimulationBox().getMolecules()[1]));
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getAtomIndices()[0], 0);
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getAtomIndices()[1], 0);
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getAtomIndices()[2], 1);
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getAtomIndices()[3], 2);
    EXPECT_EQ(_engine->getForceField().getImproperDihedrals()[0].getType(), 7);

    lineElements = {"1", "1", "2", "3", "4"};
    EXPECT_THROW(improperDihedralSection.processSection(lineElements, *_engine), customException::TopologyException);

    lineElements = {"1", "2", "7"};
    EXPECT_THROW(improperDihedralSection.processSection(lineElements, *_engine), customException::TopologyException);
}

/**
 * @brief test if endedNormally throws exception
 *
 */
TEST_F(TestTopologySection, endedNormallyImproperDihedral)
{
    readInput::topology::ImproperDihedralSection improperDihedralSection;
    EXPECT_THROW(improperDihedralSection.endedNormally(false), customException::TopologyException);
    EXPECT_NO_THROW(improperDihedralSection.endedNormally(true));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}