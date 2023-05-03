#ifndef _INPUT_FILE_READER_H_

#define _INPUT_FILE_READER_H_

#include <string>
#include <map>
#include <vector>

#include "settings.hpp"

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
        Settings &_settings;
        std::map<std::string, void (InputFileReader::*)(const std::vector<std::string> &)> _keywordMap;
        int _lineNumber = 1;

        void parseJobType(const std::vector<std::string> &);
        void parseTimestep(const std::vector<std::string> &);

        void process(const std::vector<std::string> &);

        std::vector<std::string> getLineCommands(const std::string &) const;

    public:
        InputFileReader(const std::string &, Settings &);
        ~InputFileReader() = default;

        void read();
    };
}

/**
 * @brief reads input file and sets settings
 *
 * @param filename
 * @param settings
 *
 */
void readInputFile(const std::string &, Settings &);

#endif
