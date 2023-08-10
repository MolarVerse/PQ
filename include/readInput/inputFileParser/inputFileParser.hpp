#ifndef _INPUT_FILE_PARSER_HPP_

#define _INPUT_FILE_PARSER_HPP_

#include "engine.hpp"

namespace readInput
{
    class InputFileParser;

    void checkEqualSign(const std::string_view &view, const size_t);
    void checkCommand(const std::vector<std::string> &, const size_t);
    void checkCommandArray(const std::vector<std::string> &, const size_t);
}   // namespace readInput

using ParseFunc = std::function<void(const std::vector<std::string> &, const size_t)>;

/**
 * @class InputFileParser
 *
 * @brief Base class for parsing the input file
 *
 */
class readInput::InputFileParser
{
  protected:
    engine::Engine &_engine;

    std::map<std::string, ParseFunc> _keywordFuncMap;
    std::map<std::string, bool>      _keywordRequiredMap;
    std::map<std::string, int>       _keywordCountMap;

  public:
    explicit InputFileParser(engine::Engine &engine) : _engine(engine){};

    void                                           addKeyword(const std::string &, ParseFunc, bool);
    [[nodiscard]] std::map<std::string, ParseFunc> getKeywordFuncMap() const { return _keywordFuncMap; }
    [[nodiscard]] std::map<std::string, bool>      getKeywordRequiredMap() const { return _keywordRequiredMap; }
    [[nodiscard]] std::map<std::string, int>       getKeywordCountMap() const { return _keywordCountMap; }
};

#endif   // _INPUT_FILE_PARSER_HPP_