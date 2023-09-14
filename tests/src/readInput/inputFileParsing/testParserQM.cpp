#include "exceptions.hpp"            // for InputFileException, customException
#include "inputFileParser.hpp"       // for readInput
#include "inputFileParserQM.hpp"     // for InputFileParserQM
#include "qmSettings.hpp"            // for QMSettings
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for ASSERT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for TEST_F, EXPECT_EQ, RUN_ALL_TESTS
#include <string>          // for string, allocator

using namespace readInput;

TEST_F(TestInputFileReader, parseQMMethod)
{
    EXPECT_EQ(settings::QMSettings::getQMMethod(), settings::QMMethod::NONE);

    auto parser = InputFileParserQM(*_engine);
    parser.parseQMMethod({"qm_prog", "=", "dftbplus"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMMethod(), settings::QMMethod::DFTBPLUS);

    ASSERT_THROW_MSG(parser.parseQMMethod({"qm_prog", "=", "notAMethod"}, 0),
                     customException::InputFileException,
                     "Invalid qm_prog \"notAMethod\" in input file - possible values are: dftbplus")
}

TEST_F(TestInputFileReader, parseQMScript)
{
    auto parser = InputFileParserQM(*_engine);
    parser.parseQMScript({"qm_script", "=", "script.sh"}, 0);
    EXPECT_EQ(settings::QMSettings::getQMScript(), "script.sh");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}
