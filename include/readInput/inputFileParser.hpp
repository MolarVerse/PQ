#ifndef _INPUT_FILE_PARSER_HPP_

#define _INPUT_FILE_PARSER_HPP_

#include "engine.hpp"

namespace readInput
{
    class InputFileParser;
}

class readInput::InputFileParser
{
  private:
    engine::Engine             &_engine;
    std::map<std::string, bool> _keywordRequiredMap;

  public:
    InputFileParser(engine::Engine &) : _engine(_engine){};
};

#endif   // _INPUT_FILE_PARSER_HPP_