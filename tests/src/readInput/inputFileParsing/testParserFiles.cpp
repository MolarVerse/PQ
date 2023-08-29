#include "exceptions.hpp"             // for InputFileException
#include "fileSettings.hpp"           // for FileSettings
#include "inputFileParser.hpp"        // for readInput
#include "inputFileParserFiles.hpp"   // for InputFileParserFiles
#include "testInputFileReader.hpp"    // for TestInputFileReader
#include "throwWithMessage.hpp"       // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult, testing
#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS
#include <iosfwd>          // for std
#include <string>          // for string, allocator, basic_string
#include <vector>          // for vector

using namespace std;
using namespace readInput;
using namespace ::testing;

/**
 * @brief tests parsing the "topology_file" command
 *
 * @details if the filename is empty  or does not exist it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseTopologyFilename)
{
    InputFileParserFiles parser(_engine);

    std::vector<std::string> lineElements = {"topology_file", "=", "topology.txt"};
    EXPECT_THROW_MSG(parser.parseTopologyFilename(lineElements, 0),
                     customException::InputFileException,
                     "Cannot open topology file - filename = topology.txt");

    lineElements = {"topology_file", "=", "data/topologyReader/topology.top"};
    parser.parseTopologyFilename(lineElements, 0);
    EXPECT_EQ(settings::FileSettings::getTopologyFileName(), "data/topologyReader/topology.top");
}

/**
 * @brief tests parsing the "parameter_file" command
 *
 * @details if the filename is empty or does not exist it throws inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseParameterFilename)
{
    InputFileParserFiles parser(_engine);

    std::vector<std::string> lineElements = {"parameter_file", "=", "param.txt"};
    EXPECT_THROW_MSG(parser.parseParameterFilename(lineElements, 0),
                     customException::InputFileException,
                     "Cannot open parameter file - filename = param.txt");

    lineElements = {"parameter_file", "=", "data/parameterFileReader/param.param"};
    parser.parseParameterFilename(lineElements, 0);
    EXPECT_EQ(settings::FileSettings::getParameterFilename(), "data/parameterFileReader/param.param");
}

/**
 * @brief tests parsing the intra non bonded file name
 *
 */
TEST_F(TestInputFileReader, parseIntraNonBondedFile)
{
    InputFileParserFiles     parser(_engine);
    std::vector<std::string> lineElements = {"intra-nonBonded_file", "=", "intra.dat"};
    EXPECT_THROW_MSG(parser.parseIntraNonBondedFile(lineElements, 0),
                     customException::InputFileException,
                     "Intra non bonded file \"intra.dat\" File not found");

    lineElements = {"intra-nonBonded_file", "=", "data/intraNonBondedReader/intraNonBonded.dat"};
    parser.parseIntraNonBondedFile(lineElements, 0);
    EXPECT_EQ(settings::FileSettings::getIntraNonBondedFileName(), "data/intraNonBondedReader/intraNonBonded.dat");
}

/**
 * @brief tests parsing the "start_file" command
 *
 */
TEST_F(TestInputFileReader, testStartFileName)
{
    InputFileParserFiles parser(_engine);
    vector<string>       lineElements = {"startFile_name", "=", "start.xyz"};
    EXPECT_THROW_MSG(parser.parseStartFilename(lineElements, 0),
                     customException::InputFileException,
                     "Cannot open start file - filename = start.xyz");

    lineElements = {"startFile_name", "=", "data/atomSection/testProcess.rst"};
    parser.parseStartFilename(lineElements, 0);
    EXPECT_EQ(settings::FileSettings::getStartFileName(), "data/atomSection/testProcess.rst");
}

/**
 * @brief tests parsing the "moldescriptor_file" command
 *
 */
TEST_F(TestInputFileReader, testMoldescriptorFileName)
{
    InputFileParserFiles parser(_engine);
    vector<string>       lineElements = {"moldescriptorFile_name", "=", "moldescriptor.txt"};
    EXPECT_THROW_MSG(parser.parseMoldescriptorFilename(lineElements, 0),
                     customException::InputFileException,
                     "Cannot open moldescriptor file - filename = moldescriptor.txt - file not found");

    lineElements = {"moldescriptorFile_name", "=", "data/moldescriptorReader/moldescriptor.dat"};
    parser.parseMoldescriptorFilename(lineElements, 0);
    EXPECT_EQ(settings::FileSettings::getMolDescriptorFileName(), "data/moldescriptorReader/moldescriptor.dat");
}

/**
 * @brief tests parsing the "guff_path" command
 *
 */
TEST_F(TestInputFileReader, testGuffPath)
{
    InputFileParserFiles parser(_engine);
    vector<string>       lineElements = {"guff_path", "=", "guff"};
    EXPECT_THROW_MSG(parser.parseGuffPath(lineElements, 0),
                     customException::InputFileException,
                     R"(The "guff_path" keyword id deprecated. Please use "guffdat_file" instead.)");
}

/**
 * @brief tests parsing the "guffdat_file" command
 *
 */
TEST_F(TestInputFileReader, guffDatFilename)
{
    InputFileParserFiles parser(_engine);
    vector<string>       lineElements = {"guffdat_file", "=", "guff.dat"};
    EXPECT_THROW_MSG(parser.parseGuffDatFilename(lineElements, 0),
                     customException::InputFileException,
                     "Cannot open guff file - filename = guff.dat");

    lineElements = {"guffdat_file", "=", "data/guffDatReader/guff.dat"};
    parser.parseGuffDatFilename(lineElements, 0);
    EXPECT_EQ(settings::FileSettings::getGuffDatFileName(), "data/guffDatReader/guff.dat");
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}