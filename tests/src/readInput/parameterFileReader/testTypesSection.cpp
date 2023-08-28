#include "exceptions.hpp"                 // for ParameterFileException
#include "parameterFileSection.hpp"       // for parameterFile
#include "potentialSettings.hpp"          // for PotentialSettings
#include "testParameterFileSection.hpp"   // for TestParameterFileSection
#include "throwWithMessage.hpp"           // for ASSERT_THROW_MSG
#include "typesSection.hpp"               // for TypesSection

#include "gtest/gtest.h"   // for Message, TestPartResult, tes...
#include <gtest/gtest.h>   // for EXPECT_THROW, TestInfo (ptr ...
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace ::testing;
using namespace readInput::parameterFile;

/**
 * @brief test types section processing one line
 *
 */
TEST_F(TestParameterFileSection, processSectionTypes)
{
    std::vector<std::string>               lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23", "0.99"};
    readInput::parameterFile::TypesSection typesSection;
    typesSection.process(lineElements, *_engine);
    EXPECT_EQ(settings::PotentialSettings::getScale14Coulomb(), 0.23);
    EXPECT_EQ(settings::PotentialSettings::getScale14VanDerWaals(), 0.99);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23", "1.01"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "1.23", "0.01"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "-0.23", "0.01"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);

    lineElements = {"1", "2", "1.0", "0", "s", "f", "0.23", "-0.01"};
    EXPECT_THROW(typesSection.process(lineElements, *_engine), customException::ParameterFileException);
}

TEST_F(TestParameterFileSection, endedNormallyTypes)
{
    auto typesSection = TypesSection();
    ASSERT_NO_THROW(typesSection.endedNormally(true));

    ASSERT_THROW_MSG(typesSection.endedNormally(false),
                     customException::ParameterFileException,
                     "Parameter file types section ended abnormally!");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}