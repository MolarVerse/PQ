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

#ifndef _RST_FILE_OUTPUT_HPP_

#define _RST_FILE_OUTPUT_HPP_

#include <cstddef>   // for size_t

#include "output.hpp"
#include "typeAliases.hpp"

namespace output
{
    /**
     * @class RstFileOutput inherits from Output
     *
     * @brief Output file for restart file
     *
     */
    class RstFileOutput : public Output
    {
       public:
        using Output::Output;

        void write(pq::SimBox &, const pq::Thermostat &, const size_t);
        void writeNHChain(const pq::Thermostat &, std::ostringstream &);
    };

}   // namespace output

#endif   // _RST_FILE_OUTPUT_HPP_