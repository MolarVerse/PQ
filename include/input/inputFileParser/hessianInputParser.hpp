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

#ifndef INPUT__INPUT_FILE_PARSER__HESSIAN_INPUT_PARSER_HPP
#define INPUT__INPUT_FILE_PARSER__HESSIAN_INPUT_PARSER_HPP

#include <mstd/types.hpp>

#include "inputFileParser.hpp"

class HessianSettings;

namespace input
{
    /**
     * @brief Parser for Hessian-related input file keys.
     *
     * This class is responsible for parsing Hessian-specific settings
     * from the input file and populating the corresponding HessianSettings
     * object.
     */
    class HessianInputParser : public InputFileParser
    {
        mstd::Ref<HessianSettings> _hessianSettings;

       public:
        explicit HessianInputParser(HessianSettings &hessianSettings);

        void addHessianFileKey();
        void addHessianInfoFileKey();
        void addDisplacementKey();
        void addOptimizeBeforeHessianKey();
        void addBuilderKey();
    };

}   // namespace input

#endif   // INPUT__INPUT_FILE_PARSER__HESSIAN_INPUT_PARSER_HPP
