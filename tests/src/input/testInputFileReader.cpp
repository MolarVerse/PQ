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

#include "testInputFileReader.hpp"

#include <gtest/gtest.h>   // for Message, TestPartResult

#include <fstream>   // for ofstream
#include <map>       // for map
#include <memory>    // for unique_ptr
#include <sstream>   // for basic_istringstream
#include <vector>    // for vector, _Bit_iterator, _Bit_reference

#include "exceptions.hpp"          // for InputFileException, customException
#include "gtest/gtest.h"           // for Message, TestPartResult
#include "inputFileParser.hpp"     // for readInput
#include "mmmdEngine.hpp"          // for MMMDEngine
#include "potentialSettings.hpp"   // for PotentialSettings
#include "settings.hpp"            // for Settings
#include "throwWithMessage.hpp"    // for throwWithMessage

using namespace input;
using namespace settings;

static void readKeywordList(
    const std::string        &filename,
    std::vector<std::string> &keywords,
    std::vector<bool>        &required
)
{
    std::string   line;
    std::ifstream inputFile(filename);

    while (getline(inputFile, line))
    {
        std::string keyword;
        std::string requiredString = "";
        bool        requiredBool   = false;

        if (std::istringstream(line).str().empty())
            continue;
        std::istringstream(line) >> keyword >> requiredString;
        std::istringstream(requiredString) >> std::boolalpha >> requiredBool;

        keywords.push_back(keyword);
        required.push_back(requiredBool);
    }
}

TEST_F(TestInputFileReader, testAddKeyword)
{
    std::vector<std::string> keywordsRef(0);
    std::vector<bool>        requiredRef(0);

    ::readKeywordList(
        "data/inputFileReader/keywordList.txt",
        keywordsRef,
        requiredRef
    );

    EXPECT_EQ(
        _inputFileReader->getKeywordCountMap().size(),
        keywordsRef.size()
    );
    EXPECT_EQ(
        _inputFileReader->getKeywordRequiredMap().size(),
        keywordsRef.size()
    );
    EXPECT_EQ(_inputFileReader->getKeywordFuncMap().size(), keywordsRef.size());

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        std::string keyword  = keywordsRef[i];
        bool        required = requiredRef[i];

        EXPECT_EQ(_inputFileReader->getKeywordCount(keyword), 0);
        EXPECT_FALSE(_inputFileReader->getKeywordSet(keyword));
        EXPECT_EQ(_inputFileReader->getKeywordRequired(keyword), required);
    }
}

TEST_F(TestInputFileReader, testNotAValidKeyword)
{
    auto lineElements = std::vector<std::string>{"notAValidKeyword", "=", "1"};
    ASSERT_THROW(
        _inputFileReader->process(lineElements),
        customException::InputFileException
    );
}

TEST_F(TestInputFileReader, testProcess)
{
    auto lineElements = std::vector<std::string>{"nstep", "=", "1000"};
    _inputFileReader->process(lineElements);
    EXPECT_EQ(_inputFileReader->getKeywordCount(lineElements[0]), 1);
    EXPECT_TRUE(_inputFileReader->getKeywordSet(lineElements[0]));
}

TEST_F(TestInputFileReader, testGetKeywordSetFromSetKeywordCount)
{
    const auto keyword = std::string("input_keyword");

    _inputFileReader->setKeywordCount(keyword, 0);
    EXPECT_FALSE(_inputFileReader->getKeywordSet(keyword));

    _inputFileReader->setKeywordCount(keyword, 2);
    EXPECT_TRUE(_inputFileReader->getKeywordSet(keyword));
}

TEST_F(TestInputFileReader, testRead)
{
    std::string filename = "data/inputFileReader/inputFile.txt";
    _inputFileReader_mdEngine->setFilename(filename);
    ASSERT_NO_THROW(_inputFileReader_mdEngine->read());
}

TEST_F(TestInputFileReader, testReadFileNotFound)
{
    std::string filename = "data/inputFileReader/inputFileNotFound.txt";
    _inputFileReader->setFilename(filename);
    ASSERT_THROW(_inputFileReader->read(), customException::InputFileException);
}

TEST_F(TestInputFileReader, testReadInputFileFunction)
{
    std::string filename = "data/inputFileReader/inputFile.txt";
    ASSERT_NO_THROW(readInputFile(filename, *_mdEngine));
}

TEST_F(TestInputFileReader, testReadInputFileReactionFieldMissingEpsilon)
{
    _fileName = "input_rf_missing_epsilon.in";
    {
        std::ofstream inputFile(_fileName);
        inputFile << "jobtype = mm-md;\n";
        inputFile << "integrator = v-verlet;\n";
        inputFile << "nstep = 1;\n";
        inputFile << "timestep = 0.2;\n";
        inputFile << "start_file = data/atomSection/testProcess.rst;\n";
        inputFile << "rcoulomb = 9.0;\n";
        inputFile << "long_range = reaction-field;\n";
    }

    ASSERT_THROW_MSG(
        readInputFile(_fileName, *_mdEngine),
        customException::InputFileException,
        "Missing required keyword \"rf_epsilon\" in input file: it must be "
        "set when the Coulomb long-range correction is set to "
        "\"reaction-field\"."
    );
}

TEST_F(TestInputFileReader, testReadInputFileReactionFieldWithEpsilon)
{
    _fileName = "input_rf_with_epsilon.in";
    {
        std::ofstream inputFile(_fileName);
        inputFile << "jobtype = mm-md;\n";
        inputFile << "integrator = v-verlet;\n";
        inputFile << "nstep = 1;\n";
        inputFile << "timestep = 0.2;\n";
        inputFile << "start_file = data/atomSection/testProcess.rst;\n";
        inputFile << "rcoulomb = 9.0;\n";
        inputFile << "long_range = reaction-field;\n";
        inputFile << "rf_epsilon = 80.0;\n";
    }

    ASSERT_NO_THROW(readInputFile(_fileName, *_mdEngine));
    EXPECT_EQ(
        PotentialSettings::getCoulombLongRangeType(),
        CoulombLongRangeType::REACTION_FIELD
    );
    EXPECT_EQ(PotentialSettings::getReactionFieldEpsilon(), 80.0);
}

TEST_F(TestInputFileReader, testPostProcessRequiredFail)
{
    std::vector<std::string> keywordsRef(0);
    std::vector<bool>        requiredRef(0);

    ::readKeywordList(
        "data/inputFileReader/keywordList.txt",
        keywordsRef,
        requiredRef
    );

    std::vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        std::string keyword  = keywordsRef[i];
        bool        required = requiredRef[i];

        if (required)
        {
            requiredIndex.push_back(i);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }

    for (auto const &index : requiredIndex)
    {
        const auto &keyword = keywordsRef[index];
        _inputFileReader->setKeywordCount(keyword, 0);
        ASSERT_THROW(
            _inputFileReader->postProcess(),
            customException::InputFileException
        );
        _inputFileReader->setKeywordCount(keyword, 1);
    }
}

TEST_F(TestInputFileReader, testPostProcessCountToOftenFail)
{
    std::vector<std::string> keywordsRef(0);
    std::vector<bool>        requiredRef(0);

    ::readKeywordList(
        "data/inputFileReader/keywordList.txt",
        keywordsRef,
        requiredRef
    );

    std::vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        const auto &keyword  = keywordsRef[i];
        bool        required = requiredRef[i];

        if (required)
        {
            requiredIndex.push_back(i);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }

    for (const auto &index : requiredIndex)
    {
        if (index != 1)
        {
            const auto &keyword = keywordsRef[index];
            _inputFileReader->setKeywordCount(keyword, index);
            ASSERT_THROW(
                _inputFileReader->postProcess(),
                customException::InputFileException
            );
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }
}

TEST_F(TestInputFileReader, testMoldescriptorFileProcess)
{
    std::vector<std::string> keywordsRef(0);
    std::vector<bool>        requiredRef(0);

    ::readKeywordList(
        "data/inputFileReader/keywordList.txt",
        keywordsRef,
        requiredRef
    );

    std::vector<size_t> requiredIndex(0);

    for (size_t i = 0; i < keywordsRef.size(); ++i)
    {
        const auto &keyword  = keywordsRef[i];
        bool        required = requiredRef[i];

        if (required)
        {
            requiredIndex.push_back(i);
            _inputFileReader->setKeywordCount(keyword, 1);
        }
    }

    EXPECT_NO_THROW(_inputFileReader->postProcess());
}

TEST_F(TestInputFileReader, testReadJobType)
{
    std::string filename = "data/inputFileReader/inputFile.txt";
    auto        engine   = std::unique_ptr<engine::Engine>();
    ASSERT_NO_THROW(input::readJobType(filename, engine));
    EXPECT_EQ(settings::Settings::getJobtype(), settings::JobType::MM_MD);
    EXPECT_EQ(typeid(*engine), typeid(engine::MMMDEngine));

    filename = "fileNotFound";
    ASSERT_THROW_MSG(
        input::readJobType(filename, engine),
        customException::InputFileException,
        "\"fileNotFound\" File not found"
    );

    filename = "data/inputFileReader/missingJobType.txt";
    ASSERT_THROW_MSG(
        input::readJobType(filename, engine),
        customException::InputFileException,
        "Missing keyword \"jobtype\" in input file"
    );
}