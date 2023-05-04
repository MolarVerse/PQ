#ifndef _INPUT_FILE_READER_H_

#define _INPUT_FILE_READER_H_

#include <string>
#include <map>
#include <vector>

#include "settings.hpp"
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
        Settings &_settings;

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
        void parseInfoFilename(const std::vector<std::string> &);
        void parseEnergyFilename(const std::vector<std::string> &);
        void parseTrajectoryFilename(const std::vector<std::string> &);
        void parseVelocityFilename(const std::vector<std::string> &);
        void parseRestartFilename(const std::vector<std::string> &);
        void parseChargeFilename(const std::vector<std::string> &);

        void addKeyword(const std::string &, void (InputFileReader::*)(const std::vector<std::string> &), bool);

        void process(const std::vector<std::string> &);

    public:
        InputFileReader(const std::string &, Settings &);
        ~InputFileReader() = default;

        void read();
        void postProcess();
    };

    void checkEqualSign(std::string_view, int);
    void checkCommand(const std::vector<std::string> &, int);
    void checkCommandArray(const std::vector<std::string> &, int);
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
