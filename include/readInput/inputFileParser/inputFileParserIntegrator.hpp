#ifndef _INPUT_FILE_PARSER_INTEGRATOR_HPP_

#define _INPUT_FILE_PARSER_INTEGRATOR_HPP_

#include "inputFileParser.hpp"   // for InputFileParser

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

namespace engine
{
    class Engine;
}   // namespace engine

namespace readInput
{
    /**
     * @class InputFileParserIntegrator inherits from InputFileParser
     *
     * @brief Parses the integrator commands in the input file
     *
     */
    class InputFileParserIntegrator : public InputFileParser
    {
      public:
        explicit InputFileParserIntegrator(engine::Engine &);

        void parseIntegrator(const std::vector<std::string> &, const size_t);
    };

}   // namespace readInput

#endif   // _INPUT_FILE_PARSER_INTEGRATOR_HPP_