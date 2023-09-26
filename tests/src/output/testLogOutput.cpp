#include "testLogOutput.hpp"

#include "systemInfo.hpp"   // for _AUTHOR_, _EMAIL_

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <format>          // for format
#include <iosfwd>          // for ifstream
#include <string>          // for getline, allocator, string

/**
 * @brief tests writing header to log file
 *
 */
TEST_F(TestLogOutput, writeHeader)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeHeader();
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                            _                                    ___   *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*          _                ( )                                 /'___)  *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*   _ _   (_)  ___ ___     _| | ______   _ _   ___ ___     ___ | (__    *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  ( '_`\ | |/' _ ` _ `\ /'_` |(______)/'_` )/' _ ` _ `\ /'___)| ,__)   *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  | (_) )| || ( ) ( ) |( (_| |       ( (_) || ( ) ( ) |( (___ | |      *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  | ,__/'(_)(_) (_) (_)`\__,_)       `\__, |(_) (_) (_)`\____)(_)      *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  | |                                    | |                           *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*  (_)                                    (_)                           *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, std::format("         Author: {}", sysinfo::_AUTHOR_));
    getline(file, line);
    EXPECT_EQ(line, std::format("         Email:  {}", sysinfo::_EMAIL_));
}

/**
 * @brief tests writing ended normally message to log file
 *
 */
TEST_F(TestLogOutput, writeEndedNormally)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeEndedNormally(0.1);
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         Elapsed time = 0.1 s");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                      pimd-qmcf ended normally                         *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*                                                                       *)");
    getline(file, line);
    EXPECT_EQ(line, R"(*************************************************************************)");
}

/**
 * @brief tests writing density warning to log file
 *
 */
TEST_F(TestLogOutput, writeDensityWarning)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeDensityWarning();
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "WARNING: Density and box dimensions set. Density will be ignored.");
}

/**
 * @brief tests writing initial momentum to log file
 *
 */
TEST_F(TestLogOutput, writeInitialMomentum)
{
    _logOutput->setFilename("default.out");
    _logOutput->writeInitialMomentum(0.1);
    _logOutput->close();
    std::ifstream file("default.out");
    std::string   line;
    getline(file, line);
    getline(file, line);
    EXPECT_EQ(line, "INFO:    Initial momentum = 0.1 Angstrom * amu / fs");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}