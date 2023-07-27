#include "testCommandLineArgs.hpp"

#include "exceptions.hpp"

/**
 * @brief tests detecting flags and input file name via console input
 *
 */
TEST_F(TestCommandLineArgs, detectFlags)
{
    std::vector<std::string> args = {"program", "input.in"};

    _commandLineArgs = new CommandLineArgs(args.size(), args);
    _commandLineArgs->detectFlags();
    EXPECT_EQ("input.in", _commandLineArgs->getInputFileName());

    // this part throws an exception only because no flags are implemented yet
    args             = {"program", "-i", "input.in"};
    _commandLineArgs = new CommandLineArgs(args.size(), args);
    EXPECT_THROW(_commandLineArgs->detectFlags(), customException::UserInputException);

    args             = {"program"};
    _commandLineArgs = new CommandLineArgs(args.size(), args);
    EXPECT_THROW(_commandLineArgs->detectFlags(), customException::UserInputException);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}