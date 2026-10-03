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

#include "inputFileParser.hpp"

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
        bool _resolveBuiltInSlakosPath;

       public:
        explicit QMInputParser(bool resolveBuiltInSlakosPath);
        explicit QMInputParser();

        static void parseMaceQMMethod(const std::string &);
        void        addQMMethodKey();
        void        addQMScriptKey();
        void        addQMScriptFullPathKey();
        void        addQMLoopTimeLimitKey();
        void        addDispersionKey();
        void        addRemoveNetForceKey();
        void        addMaceModelKey();
        void        addMaceModeKey();
        void        addMaceModelPathKey();
        void        addSlakosTypeKey();
        void        addSlakosPathKey();
        void        addThirdOrderKey();
        void        addHubbardDerivsKey();
        void        addXtbMethodKey();
        void        addFennolModelPathKey();
        void        addGPUPreprocessingKey();
    };

}   // namespace input

#endif   // _QM_INPUT_PARSER_HPP_
