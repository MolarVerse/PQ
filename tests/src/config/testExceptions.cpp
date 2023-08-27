#include "exceptions.hpp"         // for GuffDatException, InputFileException
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test
#include <memory>          // for allocator
#include <string_view>     // for string_view

/**
 * @brief tests throwing input file exception
 *
 */
TEST(TestExceptions, inputFileException)
{
    EXPECT_THROW_MSG(throw customException::InputFileException("test"), customException::InputFileException, "test");
}

/**
 * @brief tests throwing restart file exception
 *
 */
TEST(TestExceptions, rstFileException)
{
    EXPECT_THROW_MSG(throw customException::RstFileException("test"), customException::RstFileException, "test");
}

/**
 * @brief tests throwing user input exception
 *
 */
TEST(TestExceptions, UserInputException)
{
    EXPECT_THROW_MSG(throw customException::UserInputException("test"), customException::UserInputException, "test");
}

/**
 * @brief tests throwing mol descriptor exception
 *
 */
TEST(TestExceptions, molDescriptorException)
{
    EXPECT_THROW_MSG(throw customException::MolDescriptorException("test"), customException::MolDescriptorException, "test");
}

/**
 * @brief tests throwing user input warning
 *
 */
TEST(TestExceptions, userInputExceptionWarning)
{
    EXPECT_THROW_MSG(
        throw customException::UserInputExceptionWarning("test"), customException::UserInputExceptionWarning, "test");
}

/**
 * @brief tests throwing guff dat exception
 *
 */
TEST(TestExceptions, guffDatException)
{
    EXPECT_THROW_MSG(throw customException::GuffDatException("test"), customException::GuffDatException, "test");
}

/**
 * @brief tests throwing topology exception
 *
 */
TEST(TestExceptions, topologyException)
{
    EXPECT_THROW_MSG(throw customException::TopologyException("test"), customException::TopologyException, "test");
}

/**
 * @brief tests throwing parameter file exception
 *
 */
TEST(TestExceptions, parameterFileException)
{
    EXPECT_THROW_MSG(throw customException::ParameterFileException("test"), customException::ParameterFileException, "test");
}

/**
 * @brief tests throwing manostat exception
 *
 */
TEST(TestExceptions, manostatException)
{
    EXPECT_THROW_MSG(throw customException::ManostatException("test"), customException::ManostatException, "test");
}

/**
 * @brief tests throwing intraNonBonded exception
 *
 */
TEST(TestExceptions, intraNonBondedException)
{
    EXPECT_THROW_MSG(throw customException::IntraNonBondedException("test"), customException::IntraNonBondedException, "test");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}