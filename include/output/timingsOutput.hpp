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

#ifndef __TIMINGS_OUTPUT_HPP__

#define __TIMINGS_OUTPUT_HPP__

#include <string_view>   // for string_view

#include "output.hpp"   // for Output

namespace timings
{
    class GlobalTimer;   // forward declaration
}

namespace output
{

    /**
     * @class TimingsOutput inherits from Output
     *
     * @brief Output file for info file
     *
     */
    class TimingsOutput : public Output
    {
       public:
        using Output::Output;

        void write(timings::GlobalTimer &timer);
    };

}   // namespace output

#endif /* __TIMINGS_OUTPUT_HPP__ */