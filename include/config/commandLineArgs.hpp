#ifndef _COMMAND_LINE_ARGS_H_

#define _COMMAND_LINE_ARGS_H_

#include <string>
#include <vector>

/**
 * @class CommandLineArgs
 *
 * @brief Handles the command line arguments.
 *
 */
class CommandLineArgs
{
  private:
    int                      _argc;
    std::vector<std::string> _argv;
    std::string              _inputFileName;

  public:
    CommandLineArgs(const int argc, const std::vector<std::string> &argv) : _argc(argc), _argv(argv){};

    void detectFlags();

    std::string getInputFileName() const { return _inputFileName; }
};

#endif