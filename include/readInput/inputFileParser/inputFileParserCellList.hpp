#ifndef _INPUT_FILE_PARSER_CELL_LIST_HPP_

#define _INPUT_FILE_PARSER_CELL_LIST_HPP_

#include "engine.hpp"
#include "inputFileParser.hpp"

#include <string>
#include <vector>

namespace readInput
{
    class InputFileParserCellList;
}   // namespace readInput

/**
 * @class InputFileParserCellList inherits from InputFileParser
 *
 * @brief Parses the cell list commands in the input file
 *
 */
class readInput::InputFileParserCellList : public readInput::InputFileParser
{
  public:
    explicit InputFileParserCellList(engine::Engine &);

    void parseCellListActivated(const std::vector<std::string> &, const size_t);
    void parseNumberOfCells(const std::vector<std::string> &, const size_t);
};

#endif   // _INPUT_FILE_PARSER_CELL_LIST_HPP_
