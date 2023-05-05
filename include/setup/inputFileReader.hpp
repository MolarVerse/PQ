#ifndef _INPUT_FILE_READER_H_

#define _INPUT_FILE_READER_H_

#include <string>
#include <map>
#include <vector>

#include "engine.hpp"
#include "output.hpp"

namespace Setup::InputFileReader
{
    /**
     * @class InputFileReader
     *
     * @brief reads input file and sets settings
     *
     */
    class InputFileReader
    {
    private:
        const std::string _filename;
        Engine &_engine;

        std::map<std::string, void (InputFileReader::*)(const std::vector<std::string> &)> _keywordFuncMap;
        std::map<std::string, int> _keywordCountMap;
        std::map<std::string, bool> _keywordRequiredMap;

        int _lineNumber = 1;

        void parseJobType(const std::vector<std::string> &);

        void parseTimestep(const std::vector<std::string> &);
        void parseNumberOfSteps(const std::vector<std::string> &);

        void parseStartFilename(const std::vector<std::string> &);

        void parseOutputFreq(const std::vector<std::string> &);
        void parseLogFilename(const std::vector<std::string> &);

        void addKeyword(const std::string &, void (InputFileReader::*)(const std::vector<std::string> &), bool);

        void process(const std::vector<std::string> &);

    public:
        InputFileReader(const std::string &, Engine &);
        ~InputFileReader() = default;

        void read();
        void postProcess();
    };
}

/**
 * @brief reads input file and sets settings
 *
 * @param filename
 * @param settings
 *
 */
void readInputFile(const std::string &, Engine &);

#endif
