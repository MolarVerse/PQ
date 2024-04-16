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

#ifndef __INFO_OUTPUT_HPP__

#define __INFO_OUTPUT_HPP__

#include "output.hpp"   // for Output

#include <string_view>   // for string_view

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace output
{

    /**
     * @class InfoOutput inherits from Output
     *
     * @brief Output file for info file
     *
     */
    class InfoOutput : public Output
    {
      private:
        void writeHeader();
        void writeLeft(const double value, const std::string_view &name, const std::string_view &unit);
        void writeLeftScientific(const double value, const std::string_view &name, const std::string_view &unit);
        void writeRight(const double value, const std::string_view &name, const std::string_view &unit);

      public:
        using Output::Output;

        void write(const double simulationTime, const double loopTime, const physicalData::PhysicalData &data);
    };

}   // namespace output

#endif /* __INFO_OUTPUT_HPP__ */