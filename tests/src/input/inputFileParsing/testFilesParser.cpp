/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include <gtest/gtest.h>   // for InitGoogleTest, RUN_ALL_TESTS

#include <string>   // for string, allocator, basic_string
#include <vector>   // for vector

#include "exceptions.hpp"     // for InputFileException
#include "fileSettings.hpp"   // for FileSettings
#include "filesInputParser.hpp"
#include "testInputFileReader.hpp"   // for TestInputFileReader
#include "throwWithMessage.hpp"      // for EXPECT_THROW_MSG

using namespace input;

/**
 * @brief tests parsing the "topology_file" command
 *
 * @details if the filename is empty  or does not exist it throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseTopologyFilename)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("topology_file"));
    const auto &parseFunc = funcMap.at("topology_file");

    std::vector<std::string> lineElements = {
        "topology_file",
        "=",
        "topology.txt"
    };
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"topology.txt\" for key \"topology_file\" at line 0 in "
        "input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {"topology_file", "=", "data/topologyReader/topology.top"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getTopologyFileName(),
        "data/topologyReader/topology.top"
    );
}

/**
 * @brief tests parsing the "parameter_file" command
 *
 * @details if the filename is empty or does not exist it throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseParameterFilename)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("parameter_file"));
    const auto &parseFunc = funcMap.at("parameter_file");

    std::vector<std::string> lineElements = {
        "parameter_file",
        "=",
        "param.txt"
    };
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"param.txt\" for key \"parameter_file\" at line 0 in "
        "input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {
        "parameter_file",
        "=",
        "data/parameterFileReader/param.param"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getParameterFilename(),
        "data/parameterFileReader/param.param"
    );
}

/**
 * @brief tests parsing the intra non bonded file name
 *
 */
TEST_F(TestInputFileReader, parseIntraNonBondedFile)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("intra_nonbonded_file"));
    const auto &parseFunc = funcMap.at("intra_nonbonded_file");

    std::vector<std::string> lineElements = {
        "intra-nonBonded_file",
        "=",
        "intra.dat"
    };
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"intra.dat\" for key \"intra-nonBonded_file\" at line "
        "0 in input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {
        "intra-nonBonded_file",
        "=",
        "data/intraNonBondedReader/intraNonBonded.dat"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getIntraNonBondedFileName(),
        "data/intraNonBondedReader/intraNonBonded.dat"
    );
}

/**
 * @brief tests parsing the "start_file" command
 *
 */
TEST_F(TestInputFileReader, testStartFileName)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("start_file"));
    const auto &parseFunc = funcMap.at("start_file");

    std::vector<std::string> lineElements = {"start_file", "=", "start.xyz"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"start.xyz\" for key \"start_file\" at line 0 in "
        "input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {"start_file", "=", "data/atomSection/testProcess.rst"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getStartFileName(),
        "data/atomSection/testProcess.rst"
    );
}

/**
 * @brief tests parsing the "moldescriptor_file" command
 *
 */
TEST_F(TestInputFileReader, testMoldescriptorFileName)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("moldescriptorfile_name"));
    const auto &parseFunc = funcMap.at("moldescriptorfile_name");

    std::vector<std::string> lineElements = {
        "moldescriptorFile_name",
        "=",
        "moldescriptor.txt"
    };
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"moldescriptor.txt\" for key "
        "\"moldescriptorFile_name\" at line 0 in "
        "input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {
        "moldescriptorFile_name",
        "=",
        "data/moldescriptorReader/moldescriptor.dat"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getMolDescriptorFileName(),
        "data/moldescriptorReader/moldescriptor.dat"
    );
}

/**
 * @brief tests parsing the "guff_path" command
 *
 */
TEST_F(TestInputFileReader, testGuffPath)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("guff_path"));
    const auto                    &parseFunc    = funcMap.at("guff_path");
    const std::vector<std::string> lineElements = {"guff_path", "=", "guff"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Deprecated key 'guff_path' used at line 0.\n"
        R"(The "guff_path" keyword is deprecated. Please use "guffdat_file" instead.)"
    );
}

/**
 * @brief tests parsing the "guffdat_file" command
 *
 */
TEST_F(TestInputFileReader, guffDatFilename)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("guffdat_file"));
    const auto &parseFunc = funcMap.at("guffdat_file");

    std::vector<std::string> lineElements = {"guffdat_file", "=", "guff.dat"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"guff.dat\" for key \"guffdat_file\" at line 0 in "
        "input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {"guffdat_file", "=", "data/guffDatReader/guff.dat"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getGuffDatFileName(),
        "data/guffDatReader/guff.dat"
    );
}

/**
 * @brief tests parsing the "rpmd_start_file" command
 */
TEST_F(TestInputFileReader, testRpmdStartFileName)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("rpmd_start_file"));
    const auto &parseFunc = funcMap.at("rpmd_start_file");

    std::vector<std::string> lineElements = {
        "rpmd_start_file",
        "=",
        "rpmd_start.xyz"
    };
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"rpmd_start.xyz\" for key \"rpmd_start_file\" "
        "at line 0 in input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {
        "rpmd_start_file",
        "=",
        "data/inputFileReader/inputFile.txt"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getRingPolymerStartFileName(),
        "data/inputFileReader/inputFile.txt"
    );
}

/**
 * @brief tests parsing the "mshake_file" command
 */
TEST_F(TestInputFileReader, testMShakeFileName)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("mshake_file"));
    const auto &parseFunc = funcMap.at("mshake_file");

    std::vector<std::string> lineElements = {"mshake_file", "=", "mshake.dat"};

    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"mshake.dat\" for key \"mshake_file\" at line 0 in "
        "input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {"mshake_file", "=", "data/mshakeReader/mshake.dat"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getMShakeFileName(),
        "data/mshakeReader/mshake.dat"
    );
}

/**
 * @brief tests parsing the "dftb_file" command
 */
TEST_F(TestInputFileReader, testDFTBFileName)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("dftb_file"));
    const auto &parseFunc = funcMap.at("dftb_file");

    std::vector<std::string> lineElements = {
        "dftb_file",
        "=",
        "dftb_in.template"
    };

    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"dftb_in.template\" for key \"dftb_file\" at line 0 in "
        "input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {"dftb_file", "=", "data/dftbReader/dftb_in.template"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getDFTBFileName(),
        "data/dftbReader/dftb_in.template"
    );
}

/**
 * @brief tests parsing the "turbomole_file" command
 */
TEST_F(TestInputFileReader, testTMFileName)
{
    FilesInputParser parser(_engine->getIntraNonBonded());
    const auto       funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("turbomole_file"));
    const auto &parseFunc = funcMap.at("turbomole_file");

    std::vector<std::string> lineElements = {
        "turbomole_file",
        "=",
        "tm_define.template"
    };

    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"tm_define.template\" for key \"turbomole_file\" at "
        "line 0 in "
        "input file. Possible options are: existing file path"
    );

    clearParser(parser);

    lineElements = {
        "turbomole_file",
        "=",
        "data/turbomoleReader/tm_define.template"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings::FileSettings::getTMFileName(),
        "data/turbomoleReader/tm_define.template"
    );
}
