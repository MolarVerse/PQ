#include "testTopologyReader.hpp"

#include "exceptions.hpp"

using namespace ::testing;

/**
 * @brief tests isNeeded function
 *
 * @return true if shake is enabled
 * @return false
 */
TEST_F(TestTopologyReader, isNeeded)
{
    EXPECT_FALSE(_topologyReader->isNeeded());

    _engine->getConstraints().activate();
    EXPECT_TRUE(_topologyReader->isNeeded());
}

/**
 * @brief tests determineSection function
 *
 */
TEST_F(TestTopologyReader, determineSection)
{
    EXPECT_NO_THROW(_topologyReader->determineSection({"shake"}));
    EXPECT_THROW(_topologyReader->determineSection({"unknown"}), customException::TopologyException);
}

/**
 * @brief tests reading a topology file
 */
TEST_F(TestTopologyReader, read)
{
    EXPECT_NO_THROW(_topologyReader->read());

    _engine->getConstraints().activate();
    EXPECT_NO_THROW(_topologyReader->read());

    _topologyReader->setFilename("");
    EXPECT_THROW(_topologyReader->read(), customException::InputFileException);

    _topologyReader->setFilename("nonexistingfile.top");
    EXPECT_THROW(_topologyReader->read(), customException::InputFileException);
}

/**
 * @brief tests the readTopologyFile function
 *
 * @note this test does not check any logic, but it is here for completeness
 */
TEST_F(TestTopologyReader, readTopologyFile) { EXPECT_NO_THROW(setup::readTopologyFile("topology.top", *_engine)); }

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}