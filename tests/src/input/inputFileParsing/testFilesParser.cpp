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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "exceptions.hpp"
#include "fileSettings.hpp"
#include "filesInputParser.hpp"
#include "generalSettings.hpp"
#include "stringUtilities.hpp"
#include "testInputFileReader.hpp"
#include "throwWithMessage.hpp"

/**
 * @brief tests parsing the "topology_file" command
 *
 * @details if the filename is empty  or does not exist it throws
 * inputFileException
 *
 */
TEST_F(TestInputFileReader, testParseTopologyFilename)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
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
        "input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {"topology_file", "=", "data/topologyReader/topology.top"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getTopologyFileName(),
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
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
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
        "input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {
        "parameter_file",
        "=",
        "data/parameterFileReader/param.param"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getParameterFileName(),
        "data/parameterFileReader/param.param"
    );
}

/**
 * @brief tests parsing the intra non bonded file name
 *
 */
TEST_F(TestInputFileReader, parseIntraNonBondedFile)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
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
        "0 in input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {
        "intra-nonBonded_file",
        "=",
        "data/intraNonBondedReader/intraNonBonded.dat"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getIntraNonBondedFileName(),
        "data/intraNonBondedReader/intraNonBonded.dat"
    );
}

/**
 * @brief tests parsing the "start_file" command
 *
 */
TEST_F(TestInputFileReader, testStartFileName)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("start_file"));
    const auto &parseFunc = funcMap.at("start_file");

    std::vector<std::string> lineElements = {"start_file", "=", "start.xyz"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"start.xyz\" for key \"start_file\" at line 0 in "
        "input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {"start_file", "=", "data/atomSection/testProcess.rst"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getStartFileName(),
        "data/atomSection/testProcess.rst"
    );
}

/**
 * @brief tests parsing the "moldescriptor_file" command
 *
 */
TEST_F(TestInputFileReader, testMoldescriptorFileName)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("moldescriptor_file"));
    const auto &parseFunc = funcMap.at("moldescriptor_file");

    std::vector<std::string> lineElements = {
        "moldescriptor_file",
        "=",
        "moldescriptor.txt"
    };
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"moldescriptor.txt\" for key "
        "\"moldescriptor_file\" at line 0 in "
        "input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {
        "moldescriptor_file",
        "=",
        "data/moldescriptorReader/moldescriptor.dat"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getMolDescriptorFileName(),
        "data/moldescriptorReader/moldescriptor.dat"
    );
}

/**
 * @brief tests parsing the "guff_path" command
 *
 */
TEST_F(TestInputFileReader, testGuffPath)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
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
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("guff_file"));
    const auto &parseFunc = funcMap.at("guff_file");

    std::vector<std::string> lineElements = {"guff_file", "=", "guff.dat"};
    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"guff.dat\" for key \"guff_file\" at line 0 in "
        "input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {"guff_file", "=", "data/guffDatReader/guff.dat"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getGuffDatFileName(),
        "data/guffDatReader/guff.dat"
    );
}

/**
 * @brief tests parsing the "rpmd_start_file" command
 */
TEST_F(TestInputFileReader, testRpmdStartFileName)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
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
        "at line 0 in input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {
        "rpmd_start_file",
        "=",
        "data/inputFileReader/inputFile.txt"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getRingPolymerStartFileName(),
        "data/inputFileReader/inputFile.txt"
    );
}

/**
 * @brief tests parsing the "mshake_file" command
 */
TEST_F(TestInputFileReader, testMShakeFileName)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
    ASSERT_TRUE(funcMap.contains("mshake_file"));
    const auto &parseFunc = funcMap.at("mshake_file");

    std::vector<std::string> lineElements = {"mshake_file", "=", "mshake.dat"};

    EXPECT_THROW_MSG(
        parseFunc(lineElements, 0),
        exc::InputFileException,
        "Invalid value \"mshake.dat\" for key \"mshake_file\" at line 0 in "
        "input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {"mshake_file", "=", "data/mshakeReader/mshake.dat"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getMShakeFileName(),
        "data/mshakeReader/mshake.dat"
    );
}

/**
 * @brief tests parsing the "dftb_file" command
 */
TEST_F(TestInputFileReader, testDFTBFileName)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
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
        "input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {"dftb_file", "=", "data/dftbReader/dftb_in.template"};
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getDFTBFileName(),
        "data/dftbReader/dftb_in.template"
    );
}

/**
 * @brief tests parsing the "turbomole_file" command
 */
TEST_F(TestInputFileReader, testTMFileName)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto funcMap = parser.getKeywordFuncMap();
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
        "line 0 in input file. Value must be an existing file path."
    );

    _clearParser(parser);

    lineElements = {
        "turbomole_file",
        "=",
        "data/turbomoleReader/tm_define.template"
    };
    parseFunc(lineElements, 0);
    EXPECT_EQ(
        settings.fileSettings.getTMFileName(),
        "data/turbomoleReader/tm_define.template"
    );
}

TEST_F(TestInputFileReader, testOrcaFileName)
{
    Settings                settings;
    input::FilesInputParser parser{
        true,
        settings.fileSettings,
        settings.generalSettings
    };
    const auto  functions = parser.getKeywordFuncMap();
    const auto &parse     = functions.at("orca_file");
    EXPECT_THROW_MSG(
        parse({"orca_file", "=", "no-such-orca-template"}, 0),
        exc::InputFileException,
        "Invalid value \"no-such-orca-template\" for key \"orca_file\" at "
        "line 0 in input file. Value must be an existing file path."
    );
    _clearParser(parser);
    parse({"orca_file", "=", "data/atomSection/testProcess.rst"}, 0);
    EXPECT_EQ(
        settings.fileSettings.getOrcaFileName(),
        "data/atomSection/testProcess.rst"
    );
    input::FilesInputParser portable{
        false,
        settings.fileSettings,
        settings.generalSettings
    };
    portable.getKeywordFuncMap().at("orca_file")(
        {"orca_file", "=", "no-such-orca-template"},
        0
    );
    EXPECT_EQ(settings.fileSettings.getOrcaFileName(), "no-such-orca-template");
}

/**
 * @brief without path validation every file keyword takes any name, also one
 * that does not exist, and stores it unchanged
 */
TEST_F(TestInputFileReader, fileKeywordsWithoutPathValidationAcceptAnyName)
{
    struct Case
    {
        std::string keyword;
        std::string (*getter)(const FileSettings &);
    };

    const std::vector<Case> cases = {
        Case{
            .keyword = "intra-nonBonded_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getIntraNonBondedFileName(); }
        },
        Case{
            .keyword = "topology_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getTopologyFileName(); }
        },
        Case{
            .keyword = "parameter_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getParameterFileName(); }
        },
        Case{
            .keyword = "start_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getStartFileName(); }
        },
        Case{
            .keyword = "rpmd_start_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getRingPolymerStartFileName(); }
        },
        Case{
            .keyword = "moldescriptor_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getMolDescriptorFileName(); }
        },
        Case{
            .keyword = "guff_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getGuffDatFileName(); }
        },
        Case{
            .keyword = "mshake_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getMShakeFileName(); }
        },
        Case{
            .keyword = "dftb_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getDFTBFileName(); }
        },
        Case{
            .keyword = "turbomole_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getTMFileName(); }
        },
        Case{
            .keyword = "orca_file",
            .getter  = [](const FileSettings &settings)
            { return settings.getOrcaFileName(); }
        },
    };

    for (const auto &testCase : cases)
    {
        Settings settings;
        settings.generalSettings.deactivateIntraNonBondedByUser();

        input::FilesInputParser parser{
            false,
            settings.fileSettings,
            settings.generalSettings
        };
        const auto funcMap = parser.getKeywordFuncMap();
        const auto mapKey =
            utilities::toLowerAndReplaceDashesCopy(testCase.keyword);
        ASSERT_TRUE(funcMap.contains(mapKey)) << testCase.keyword;

        const std::string name = "does_not_exist_" + mapKey + ".txt";
        funcMap.at(mapKey)({testCase.keyword, "=", name}, 0);

        EXPECT_EQ(testCase.getter(settings.fileSettings), name)
            << testCase.keyword;
        EXPECT_EQ(
            settings.generalSettings.isIntraNonBondedActivated(),
            testCase.keyword == "intra-nonBonded_file"
        ) << testCase.keyword;
    }
}

/**
 * @brief the intra non-bonded keyword activates the intra non-bonded
 * interactions in both modes, with an existing file and with a plain name
 */
TEST_F(TestInputFileReader, intraNonBondedKeywordActivatesInteractions)
{
    for (const auto validate : {true, false})
    {
        Settings                settings;
        input::FilesInputParser parser{
            validate,
            settings.fileSettings,
            settings.generalSettings
        };
        const auto  funcMap   = parser.getKeywordFuncMap();
        const auto &parseFunc = funcMap.at("intra_nonbonded_file");

        parseFunc(
            {"intra-nonBonded_file",
             "=",
             "data/intraNonBondedReader/intraNonBonded.dat"},
            0
        );

        EXPECT_TRUE(settings.generalSettings.isIntraNonBondedActivated())
            << validate;
        EXPECT_EQ(
            settings.fileSettings.getIntraNonBondedFileName(),
            "data/intraNonBondedReader/intraNonBonded.dat"
        ) << validate;
    }
}
