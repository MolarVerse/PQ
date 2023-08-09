#ifndef _INPUT_FILE_READER_HPP_

#define _INPUT_FILE_READER_HPP_

#include "engine.hpp"
#include "exceptions.hpp"
#include "inputFileParser.hpp"
#include "output.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief namespace for reading input files
 *
 */
namespace readInput
{
    class InputFileReader;
    void readInputFile(const std::string &, engine::Engine &);
}   // namespace readInput

using ParseFunc = std::function<void(const std::vector<std::string> &, const size_t)>;

/**
 * @class InputFileReader
 *
 * @brief reads input file and sets settings
 *
 */
class readInput::InputFileReader
{
  private:
    std::string     _filename;
    engine::Engine &_engine;

    std::map<std::string, ParseFunc> _keywordFuncMap;
    std::map<std::string, int>       _keywordCountMap;
    std::map<std::string, bool>      _keywordRequiredMap;

    std::vector<std::unique_ptr<InputFileParser>> _parsers;

    size_t _lineNumber = 1;

  public:
    InputFileReader(const std::string &filename, engine::Engine &engine);

    void read();
    void addKeywords();
    void process(const std::vector<std::string> &);
    void postProcess();

    /********************************
     *                              *
     * standard getters and setters *
     *                              *
     ********************************/

    void setFilename(const std::string_view filename) { _filename = filename; }
    void setKeywordCount(const std::string &keyword, const int count) { _keywordCountMap[keyword] = count; }

    [[nodiscard]] int  getKeywordCount(const std::string &keyword) { return _keywordCountMap[keyword]; }
    [[nodiscard]] bool getKeywordRequired(const std::string &keyword) { return _keywordRequiredMap[keyword]; }

    [[nodiscard]] std::map<std::string, int>       getKeywordCountMap() const { return _keywordCountMap; }
    [[nodiscard]] std::map<std::string, bool>      getKeywordRequiredMap() const { return _keywordRequiredMap; }
    [[nodiscard]] std::map<std::string, ParseFunc> getKeywordFuncMap() const { return _keywordFuncMap; }
};

#endif   // _INPUT_FILE_READER_HPP_
