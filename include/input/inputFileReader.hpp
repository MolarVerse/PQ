/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef INPUT__INPUT_FILE_READER_HPP
#define INPUT__INPUT_FILE_READER_HPP

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "enums/jobtype.hpp"
#include "inputFileParser.hpp"

namespace engine
{
    class Engine;   // forward declaration
}   // namespace engine

struct Settings;               // forward declaration
class SimulationBoxSettings;   // forward declaration
class QMSettings;              // forward declaration
class ManostatSettings;        // forward declaration

/**
 * @brief namespace for reading input files
 *
 */
namespace input
{
    [[nodiscard]]
    JobType readJobType(const std::string &fileName);
    void    readInputFile(const std::string_view &fileName, Settings &settings);
    void    processEqualSign(std::string &command, size_t lineNumber);

    /**
     * @class InputFileReader
     *
     * @brief reads input file and sets settings
     *
     */
    class InputFileReader
    {
       private:
        std::string _fileName;

        std::map<std::string, InputFileParser::ParseFunc> _keywordFuncMap;
        std::map<std::string, size_t>                     _keywordCountMap;
        std::map<std::string, bool>                       _keywordRequiredMap;
        std::map<std::string, bool>                       _keywordSetMap;

        std::vector<std::unique_ptr<InputFileParser>> _parsers;

        size_t _lineNumber = 1;

       public:
        explicit InputFileReader(
            const std::string_view &fileName,
            Settings               &settings,
            bool                    validateFilePaths,
            bool                    resolveBuiltInSlakosPath
        );
        explicit InputFileReader(
            const std::string_view &fileName,
            Settings               &settings
        );

        void read();
        void addKeywords();
        void process(const std::vector<std::string> &lineElements);
        void postProcess();
        void validateInputConfiguration(const Settings &settings) const;

        /***************************
         * standard setter methods *
         ***************************/

        void setFilename(std::string_view fileName);
        void setKeywordCount(const std::string &keyword, size_t count);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getKeywordCount(const std::string &keyword) const;
        [[nodiscard]] bool   getKeywordSet(const std::string &keyword) const;
        [[nodiscard]] bool getKeywordRequired(const std::string &keyword) const;

        [[nodiscard]] std::map<std::string, size_t> getKeywordCountMap() const;
        [[nodiscard]] std::map<std::string, bool>   getKeywordSetMap() const;
        [[nodiscard]] std::map<std::string, bool> getKeywordRequiredMap() const;
        [[nodiscard]] std::map<std::string, InputFileParser::ParseFunc> getKeywordFuncMap(
        ) const;

       private:
        /******************************
         * input validation functions *
         ******************************/

        void _validateTimings(const Settings &settings) const;
        void _validateQM(const QMSettings &qmSettings) const;
        void _validateThermostat(
            const SimulationBoxSettings &simBoxSettings
        ) const;
        void _validateManostat(const ManostatSettings &manostatSettings) const;
        void _validateReactionFieldCoulomb() const;
        void _validateRingPolymer() const;
        static void _validateCellList();
    };

}   // namespace input

#endif   // INPUT__INPUT_FILE_READER_HPP
