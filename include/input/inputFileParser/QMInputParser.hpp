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

#ifndef _QM_INPUT_PARSER_HPP_

#define _QM_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t

#include "inputFileParser.hpp"   // for InputFileParser
#include "logOutput.hpp"

namespace input
{
    /**
     * @brief QMInputParser inherits from InputFileParser
     *
     * @details Parses the general commands in the input file
     *
     */
    class QMInputParser : public InputFileParser
    {
       private:
        out::LogOutput *_logOutput;

        bool _resolveBuiltInSlakosPath;

       public:
        explicit QMInputParser(out::LogOutput &, bool resolveBuiltInSlakosPath);
        explicit QMInputParser(out::LogOutput &);

        static void parseQMMethod(const std::vector<std::string> &, size_t);
        static void parseQMScript(const std::vector<std::string> &, size_t);
        static void parseQMScriptFullPath(
            const std::vector<std::string> &,
            size_t
        );
        static void parseQMLoopTimeLimit(
            const std::vector<std::string> &,
            size_t
        );

        static void parseDispersion(const std::vector<std::string> &, size_t);
        static void parseRemoveNetForce(
            const std::vector<std::string> &,
            size_t
        );

        void        parseMaceModel(const std::vector<std::string> &, size_t);
        static void parseMaceMode(const std::vector<std::string> &, size_t);
        static void parseMaceModelPath(
            const std::vector<std::string> &,
            size_t
        );
        static void parseMaceQMMethod(const std::string_view &);

        void parseSlakosType(const std::vector<std::string> &, size_t) const;
        static void parseSlakosPath(const std::vector<std::string> &, size_t);
        static void parseThirdOrder(const std::vector<std::string> &, size_t);
        static void parseHubbardDerivs(
            const std::vector<std::string> &,
            size_t
        );

        static void parseXtbMethod(const std::vector<std::string> &, size_t);

        static void parseFennolModelPath(
            const std::vector<std::string> &,
            size_t
        );
        static void parseGPUPreprocessing(
            const std::vector<std::string> &,
            size_t
        );
    };

}   // namespace input

#endif   // _QM_INPUT_PARSER_HPP_
