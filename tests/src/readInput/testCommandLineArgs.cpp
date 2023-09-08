#include "../../../include/readInput/commandLineArgs.hpp"    // for CommandLineArgs
#include "exceptions.hpp"         // for UserInputException
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), InitGoogl...
#include <string>          // for allocator, basic_string, operator+
#include <vector>          // for vector

/**
 * @brief tests detecting flags and input file name via console input
 *
 */
TEST(TestCommandLineArgs, detectFlags)
{
    std::vector<std::string> args            = {"program", "input.in"};
    auto                     commandLineArgs = CommandLineArgs(int(args.size()), args);

    commandLineArgs.detectFlags();
    EXPECT_EQ("input.in", commandLineArgs.getInputFileName());
}

/**
 * @brief tests detecting flags and input file name via console input
 *
 * @TODO: no flags implemented at the moment
 */
TEST(TestCommandLineArgs, detectFlags_flag_given)
{
    std::vector<std::string> args            = {"program", "-i", "input.in"};
    auto                     commandLineArgs = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(commandLineArgs.detectFlags(),
                     customException::UserInputException,
                     "Invalid flag: " + args[1] + " Flags are not yet implemented.");
}

/**
 * @brief tests throwing exception if no input file name is given
 *
 */
TEST(TestCommandLineArgs, detectFlags_missing_input_file)
{
    std::vector<std::string> args            = {"program"};
    auto                     commandLineArgs = CommandLineArgs(int(args.size()), args);

    EXPECT_THROW_MSG(commandLineArgs.detectFlags(),
                     customException::UserInputException,
                     "No input file specified. Usage: pimd_qmcf <input_file>");
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}