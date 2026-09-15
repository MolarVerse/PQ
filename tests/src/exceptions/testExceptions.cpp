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

#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), TEST, InitGoogleTest, RUN_ALL_TESTS

#include <optional>      // for optional
#include <string_view>   // for string_view

#include "exceptions.hpp"         // for GuffDatException, InputFileException
#include "gtest/gtest.h"          // for Message, TestPartResult
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

/**
 * @brief tests throwing input file exception
 *
 */
TEST(TestExceptions, inputFileException)
{
    EXPECT_THROW_MSG(
        throw exc::InputFileException("test"),
        exc::InputFileException,
        "test"
    );
}

/**
 * @brief tests structured source line metadata
 */
TEST(TestExceptions, sourceLine)
{
    auto exception = exc::InputFileException("test", 12);

    EXPECT_EQ(exception.getLineNumber(), std::optional<size_t>(12));

    exception.setLineNumber(13);
    EXPECT_EQ(exception.getLineNumber(), std::optional<size_t>(12));
}

/**
 * @brief tests throwing restart file exception
 *
 */
TEST(TestExceptions, rstFileException)
{
    EXPECT_THROW_MSG(
        throw exc::RstFileException("test"),
        exc::RstFileException,
        "test"
    );
}

/**
 * @brief tests throwing user input exception
 *
 */
TEST(TestExceptions, UserInputException)
{
    EXPECT_THROW_MSG(
        throw exc::UserInputException("test"),
        exc::UserInputException,
        "test"
    );
}

/**
 * @brief tests throwing mol descriptor exception
 *
 */
TEST(TestExceptions, molDescriptorException)
{
    EXPECT_THROW_MSG(
        throw exc::MolDescriptorException("test"),
        exc::MolDescriptorException,
        "test"
    );
}

/**
 * @brief tests throwing user input warning
 *
 */
TEST(TestExceptions, userInputExceptionWarning)
{
    EXPECT_THROW_MSG(
        throw exc::UserInputExceptionWarning("test"),
        exc::UserInputExceptionWarning,
        "test"
    );
}

/**
 * @brief tests throwing guff dat exception
 *
 */
TEST(TestExceptions, guffDatException)
{
    EXPECT_THROW_MSG(
        throw exc::GuffDatException("test"),
        exc::GuffDatException,
        "test"
    );
}

/**
 * @brief tests throwing topology exception
 *
 */
TEST(TestExceptions, topologyException)
{
    EXPECT_THROW_MSG(
        throw exc::TopologyException("test"),
        exc::TopologyException,
        "test"
    );
}

/**
 * @brief tests throwing parameter file exception
 *
 */
TEST(TestExceptions, parameterFileException)
{
    EXPECT_THROW_MSG(
        throw exc::ParameterFileException("test"),
        exc::ParameterFileException,
        "test"
    );
}

/**
 * @brief tests throwing manostat exception
 *
 */
TEST(TestExceptions, manostatException)
{
    EXPECT_THROW_MSG(
        throw exc::ManostatException("test"),
        exc::ManostatException,
        "test"
    );
}

/**
 * @brief tests throwing intraNonBonded exception
 *
 */
TEST(TestExceptions, intraNonBondedException)
{
    EXPECT_THROW_MSG(
        throw exc::IntraNonBondedException("test"),
        exc::IntraNonBondedException,
        "test"
    );
}

/**
 * @brief tests throwing intraNonBonded exception
 *
 */
TEST(TestExceptions, shakeException)
{
    EXPECT_THROW_MSG(
        throw exc::ShakeException("test"),
        exc::ShakeException,
        "test"
    );
}

/**
 * @brief tests throwing intraNonBonded exception
 *
 */
TEST(TestExceptions, cellListException)
{
    EXPECT_THROW_MSG(
        throw exc::CellListException("test"),
        exc::CellListException,
        "test"
    );
}

/**
 * @brief tests throwing ring polymer restart file exception
 *
 */
TEST(TestExceptions, ringPolymerRestartFileException)
{
    EXPECT_THROW_MSG(
        throw exc::RingPolymerRestartFileException("test"),
        exc::RingPolymerRestartFileException,
        "test"
    );
}

/**
 * @brief tests throwing qm runner exception
 *
 */
TEST(TestExceptions, qmRunnerException)
{
    EXPECT_THROW_MSG(
        throw exc::QMRunnerException("test"),
        exc::QMRunnerException,
        "test"
    );
}

TEST(TestExceptions, hybridConfiguratorException)
{
    EXPECT_THROW_MSG(
        throw exc::HybridConfiguratorException("test"),
        exc::HybridConfiguratorException,
        "test"
    );
}

TEST(TestExceptions, hybridMDEngineException)
{
    EXPECT_THROW_MSG(
        throw exc::HybridMDEngineException("test"),
        exc::HybridMDEngineException,
        "test"
    );
}
