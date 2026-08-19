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

#ifndef _FILES_INPUT_PARSER_HPP_

#define _FILES_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t

#include "inputFileParser.hpp"   // for InputFileParser

namespace input
{
    /**
     * @class FilesInputParser inherits from InputFileParser
     *
     * @brief Parses all input file commands related to input files
     *
     */
    class FilesInputParser : public InputFileParser
    {
       private:
        std::shared_ptr<intraNonBonded::IntraNonBonded> _intraNonBonded;

        bool _validateFilePaths;

       public:
        explicit FilesInputParser(
            engine::Engine &,
            std::shared_ptr<intraNonBonded::IntraNonBonded> intraNonBonded,
            bool                                            validateFilePaths
        );
        explicit FilesInputParser(
            engine::Engine &,
            std::shared_ptr<intraNonBonded::IntraNonBonded> intraNonBonded
        );

        void parseIntraNonBondedFile(
            const std::vector<std::string> &,
            const size_t
        );

        void parseTopologyFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseParameterFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseStartFilename(const std::vector<std::string> &, const size_t);

        void parseRingPolymerStartFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseMoldescriptorFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseGuffDatFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseGuffPath(const std::vector<std::string> &, const size_t);

        void parseMShakeFilename(
            const std::vector<std::string> &,
            const size_t
        );

        void parseDFTBFilename(const std::vector<std::string> &, const size_t);

        void parseTMFilename(const std::vector<std::string> &, const size_t);
    };

}   // namespace input

#endif   // _FILES_INPUT_PARSER_HPP_
