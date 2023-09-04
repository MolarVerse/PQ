#include "testTopologyReader.hpp"

#include "constraints.hpp"       // for Constraints
#include "exceptions.hpp"        // for InputFileException, TopologyException
#include "fileSettings.hpp"      // for FileSettings
#include "forceFieldClass.hpp"   // for ForceField

#include "gtest/gtest.h"   // for AssertionResult, Message, TestPartResult

/**
 * @brief tests isNeeded function
 *
 * @return true if shake is enabled
 * @return true if forceField is enabled
 * @return false
 */
TEST_F(TestTopologyReader, isNeeded)
{
    EXPECT_FALSE(_topologyReader->isNeeded());

    _engine->getConstraints().activate();
    EXPECT_TRUE(_topologyReader->isNeeded());

    _engine->getConstraints().deactivate();
    _engine->getForceField().activate();
    EXPECT_TRUE(_topologyReader->isNeeded());
}

/**
 * @brief tests determineSection function
 *
 */
TEST_F(TestTopologyReader, determineSection)
{
    EXPECT_NO_THROW([[maybe_unused]] const auto dummy = _topologyReader->determineSection({"shake"}));
    EXPECT_THROW([[maybe_unused]] const auto dummy = _topologyReader->determineSection({"unknown"}),
                 customException::TopologyException);
}

/**
 * @brief tests reading a topology file
 */
TEST_F(TestTopologyReader, read)
{
    EXPECT_NO_THROW(_topologyReader->read());

    _engine->getConstraints().activate();
    EXPECT_NO_THROW(_topologyReader->read());

    settings::FileSettings::unsetIsTopologyFileNameSet();
    EXPECT_THROW(_topologyReader->read(), customException::InputFileException);
}

/**
 * @brief tests the readTopologyFile function
 *
 * @note this test does not check any logic, but it is here for completeness
 */
TEST_F(TestTopologyReader, readTopologyFile)
{
    settings::FileSettings::setTopologyFileName("topology.top");
    readInput::topology::readTopologyFile(*_engine);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}