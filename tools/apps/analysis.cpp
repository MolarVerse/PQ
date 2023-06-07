#include <iostream>

#include "engine.hpp"
#include "commandLineArgs.hpp"

using namespace std;

int main(int argc, char **argv)
{
    // like in main.cpp of pimd_qmcf not best way TODO:
    vector<string> arguments(argv, argv + argc);
    auto commandLineArgs = CommandLineArgs(argc, arguments);
    commandLineArgs.detectFlags();

    std::cout << "Hello, world!" << std::endl;

    auto engine = Engine(argv[0], argv[1]);

    return 0;
}