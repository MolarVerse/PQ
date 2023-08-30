#include "constants.hpp"                  // for _DEG_TO_RAD_
#include "dihedralType.hpp"               // for DihedralType
#include "engine.hpp"                     // for Engine
#include "exceptions.hpp"                 // for ParameterFileException
#include "forceField.hpp"                 // for ForceField
#include "improperDihedralSection.hpp"    // for ImproperDihedralSection
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
TEST_F(TestParameterFileSection, processSectionImproperDihedral)
{
    std::vector<std::string> lineElements = {"0", "1.22", "234.3", "324.3"};
    ImproperDihedralSection  improperDihedralSection;
    improperDihedralSection.processSection(lineElements, *_engine);
    EXPECT_EQ(_engine->getForceField().getImproperDihedralTypes().size(), 1);
    EXPECT_EQ(_engine->getForceField().getImproperDihedralTypes()[0].getId(), 0);
    EXPECT_EQ(_engine->getForceField().getImproperDihedralTypes()[0].getForceConstant(), 1.22);
    EXPECT_EQ(_engine->getForceField().getImproperDihedralTypes()[0].getPeriodicity(), 234.3);
    EXPECT_EQ(_engine->getForceField().getImproperDihedralTypes()[0].getPhaseShift(), 324.3 * constants::_DEG_TO_RAD_);

    lineElements = {"1", "2", "1.0", "0", "2"};
    EXPECT_THROW(improperDihedralSection.processSection(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "-1.0", "3"};
    EXPECT_THROW(improperDihedralSection.processSection(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyDihedral)
{
    auto improperDihedralSection = ImproperDihedralSection();
    ASSERT_NO_THROW(improperDihedralSection.endedNormally(true));

    ASSERT_THROW_MSG(improperDihedralSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file impropers section ended abnormally!");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}