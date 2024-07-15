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

#ifndef _THERMOSTAT_INPUT_PARSER_HPP_

#define _THERMOSTAT_INPUT_PARSER_HPP_

#include <cstddef>   // for size_t
#include <string>    // for string
#include <vector>    // for vector

#include "inputFileParser.hpp"   // for InputFileParser
#include "typeAliases.hpp"       // for pq::strings

namespace input
{
    /**
     * @class ThermostatInputParser inherits from InputFileParser
     *
     * @brief Parses the thermostat commands in the input file
     *
     */
    class ThermostatInputParser : public InputFileParser
    {
       public:
        explicit ThermostatInputParser(pq::Engine &);

        void parseThermostat(const pq::strings &, const size_t);
        void parseTemperature(const pq::strings &, const size_t);
        void parseStartTemperature(const pq::strings &, const size_t);
        void parseEndTemperature(const pq::strings &, const size_t);
        void parseTemperatureRampSteps(const pq::strings &, const size_t);
        void parseTemperatureRampFrequency(const pq::strings &, const size_t);
        void parseThermostatRelaxationTime(const pq::strings &, const size_t);
        void parseThermostatFriction(const pq::strings &, const size_t);
        void parseThermostatChainLength(const pq::strings &, const size_t);
        void parseThermostatCouplingFrequency(
            const pq::strings &,
            const size_t
        );
    };

}   // namespace input

#endif   // _THERMOSTAT_INPUT_PARSER_HPP_