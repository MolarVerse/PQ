#ifndef _INPUT_FILE_PARSER_PARAMETER_FILE_HPP_

#define _INPUT_FILE_PARSER_PARAMETER_FILE_HPP_

#include "inputFileParser.hpp"

#include <cstddef>   // for size_t
#include <string>
#include <vector>

namespace engine
{
    class Engine;
}   // namespace engine

namespace readInput
{
    /**
     * @class InputFileParserParameterFile inherits from InputFileParser
     *
     * @brief Parses the parameter file commands in the input file
     *
     */
    class InputFileParserParameterFile : public InputFileParser
    {
      public:
        explicit InputFileParserParameterFile(engine::Engine &);

        void parseParameterFilename(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_PARAMETER_FILE_HPP_