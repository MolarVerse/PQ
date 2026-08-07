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
#include "typeAliases.hpp"       // for std::vector<std::string>

namespace input
{
    /**
     * @class QMInputParser inherits from InputFileParser
     *
     * @brief Parses the general commands in the input file
     *
     */
    class QMInputParser : public InputFileParser
    {
       private:
        bool _resolveBuiltInSlakosPath;

       public:
        explicit QMInputParser(
            pq::Engine &,
            bool resolveBuiltInSlakosPath = true
        );

        void parseQMMethod(const std::vector<std::string> &, const size_t);
        void parseQMScript(const std::vector<std::string> &, const size_t);
        void parseQMScriptFullPath(
            const std::vector<std::string> &,
            const size_t
        );
        void parseQMLoopTimeLimit(
            const std::vector<std::string> &,
            const size_t
        );

        void parseDispersion(const std::vector<std::string> &, const size_t);
        void parseRemoveNetForce(
            const std::vector<std::string> &,
            const size_t
        );

        void parseMaceModel(const std::vector<std::string> &, const size_t);
        void parseMaceMode(const std::vector<std::string> &, const size_t);
        void parseMaceModelPath(const std::vector<std::string> &, const size_t);
        void parseMaceQMMethod(const std::string_view &);

        void parseSlakosType(const std::vector<std::string> &, const size_t);
        void parseSlakosPath(const std::vector<std::string> &, const size_t);
        void parseThirdOrder(const std::vector<std::string> &, const size_t);
        void parseHubbardDerivs(const std::vector<std::string> &, const size_t);

        void parseXtbMethod(const std::vector<std::string> &, const size_t);

        void parseFennolModelPath(
            const std::vector<std::string> &,
            const size_t
        );
        void parseGPUPreprocessing(
            const std::vector<std::string> &,
            const size_t
        );
    };

}   // namespace input

#endif   // _QM_INPUT_PARSER_HPP_
