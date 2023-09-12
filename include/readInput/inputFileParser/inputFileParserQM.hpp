#ifndef _INPUT_FILE_PARSER_QM_HPP_

#define _INPUT_FILE_PARSER_QM_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

#include <cstddef>   // for size_t
#include <memory>    // for unique_ptr
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;   // Forward declaration
}

namespace readInput
{
    /**
     * @class InputFileParserQM inherits from InputFileParser
     *
     * @brief Parses the general commands in the input file
     *
     */
    class InputFileParserQM : public InputFileParser
    {
      public:
        explicit InputFileParserQM(engine::Engine &);

        void parseQMMethod(const std::vector<std::string> &lineElements, const size_t lineNumber);
        void parseQMScript(const std::vector<std::string> &lineElements, const size_t lineNumber);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_QM_HPP_