#include "bondSection.hpp"                // for BondSection
#include "bondType.hpp"                   // for BondType
#include "engine.hpp"                     // for Engine
#include "exceptions.hpp"                 // for ParameterFileException
#include "forceFieldClass.hpp"            // for ForceField
#include "parameterFileSection.hpp"       // for parameterFile
#include "testParameterFileSection.hpp"   // for TestParameterFileSection
#include "throwWithMessage.hpp"           // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for EXPECT_EQ, TestInfo (ptr only)
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace readInput::parameterFile;

/**
 * @brief test bonds section processing one line
 *
 */
TEST_F(TestParameterFileSection, processSectionBonds)
{
    std::vector<std::string>              lineElements = {"0", "1.22", "234.3"};
    readInput::parameterFile::BondSection bondSection;
    bondSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getBondTypes().size(), 1);
    EXPECT_EQ(_engine->getForceField().getBondTypes()[0].getId(), 0);
    EXPECT_EQ(_engine->getForceField().getBondTypes()[0].getEquilibriumBondLength(), 1.22);
    EXPECT_EQ(_engine->getForceField().getBondTypes()[0].getForceConstant(), 234.3);

    lineElements = {"1", "2", "1.0", "0"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "-2", "1.0"};
    EXPECT_THROW(bondSection.processSection(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyBonds)
{
    auto bondSection = BondSection();
    ASSERT_NO_THROW(bondSection.endedNormally(true));

    ASSERT_THROW_MSG(bondSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file bonds section ended abnormally!");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}